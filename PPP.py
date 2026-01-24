import numpy as np
import torch
from vllm import LLM, SamplingParams
import argparse
import json
import tqdm
import os
import glob
import gc
import re


_RAG_SYSTEM_PROMPT = """You are a helpful assistant designed to generate personalized responses to user questions. Your task is to answer a user's question from a post in a personalized way by considering this user's past post questions and detailed descriptions of these questions.
# Your input:
    - The user's current question from a post.
    - The user's past post questions and detailed descriptions of these questions.
# Your task: Answer the user's current question in a personalized way by considering this user's past post questions and detailed descriptions of these questions, to learn about the user's preferences.
# Your output: You should generate personalized answer to the user's current question by considering this user's past post questions and detailed descriptions of these questions to learn about user's preferences.
"""

_RAG_USER_PROMPT = """
# Past post questions and detailed descriptions of these questions:
{profile}
# Current post question:
{question}

Now please generate a personalized answer to the user's current question by considering this user's past post questions and detailed descriptions of these questions to learn about user's preferences.
"""

_REGULAR_QA_SYSTEM_PROMPT = """You are a helpful assistant designed to generate personalized responses to user questions. Your task is to answer a user's question from a post in a personalized way.

# Your input:
    - The user's current question from a post.

# Your task: Answer the user's current question in a personalized way.
# Your output: You should generate a personalized answer to the user's current question.
"""

_REGULAR_QA_USER_PROMPT = """
# Current post question:
{question}

Now please generate a personalized answer to the user's current question.
"""


import re

def get_pii_entities(text):
    """
    Identifies potential PII in the input text using simple regex heuristics.
    Returns a list of dicts: {"type": ..., "value": ...}
    """

    patterns = {
        "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "phone": r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        "url": r'https?://[^\s]+',
        "ssn_us": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b(?:\d[ -]*?){13,16}\b',
        "date_of_birth": r'\b(?:19|20)\d{2}[-/\.](?:0[1-9]|1[0-2])[-/\.](?:0[1-9]|[12]\d|3[01])\b',
    }

    entities = []

    for pii_type, pattern in patterns.items():
        for match in re.findall(pattern, text):
            entities.append({
                "type": pii_type,
                "value": match
            })

    return entities


def filter_pii_from_text(text, pii_list):
    """Removes identified PII from the generated text before it goes to Cloud."""
    filtered_text = text
    for entity in pii_list:
        # Case-insensitive replacement with a mask
        filtered_text = re.sub(re.escape(entity), "[PII]", filtered_text, flags=re.IGNORECASE)
    return filtered_text

def create_dual_prompts(data_sample, tokenizer, num_contexts=3, llm_type="qwen"):
    question = data_sample['question']
    
    if 'profile' in data_sample and data_sample['profile']:
        profile_text = "\n\n".join([item['text'] for item in data_sample['profile'][:num_contexts]])
    else:
        profile_text = "No history available."

    if llm_type == "qwen":
        public_messages = [
            {"role": "system", "content": _REGULAR_QA_SYSTEM_PROMPT},
            {"role": "user", "content": _REGULAR_QA_USER_PROMPT.format(question=question)}
        ]
    elif llm_type == "gemma":
        public_messages = [
            {"role": "user", "content":_REGULAR_QA_SYSTEM_PROMPT + "\n" +_REGULAR_QA_USER_PROMPT.format(question=question)}
        ]
    else:
        raise ValueError(f"Unsupported llm_type: {llm_type}")
    
    if llm_type == "qwen":
        private_messages = [
            {"role": "system", "content": _RAG_SYSTEM_PROMPT},
            {"role": "user", "content": _RAG_USER_PROMPT.format(profile=profile_text, question=question)}
        ]   
    elif llm_type == "gemma":
        private_messages = [
            {"role": "user", "content": _RAG_USER_PROMPT.format(profile=profile_text, question=question)}
        ]
    else:
        raise ValueError(f"Unsupported llm_type: {llm_type}")

    public_prompt_str = tokenizer.apply_chat_template(public_messages, tokenize=False, add_generation_prompt=True)
    private_prompt_str = tokenizer.apply_chat_template(private_messages, tokenize=False, add_generation_prompt=True)

    return public_prompt_str, private_prompt_str, profile_text


def get_k_tokens_cloud(llm, current_text, k):
    params = SamplingParams(max_tokens=k, temperature=0.1) 
    outputs = llm.generate(prompts=[current_text], sampling_params=params, use_tqdm=False)
    draft_ids = list(outputs[0].outputs[0].token_ids)
    draft_text = outputs[0].outputs[0].text
    return draft_ids, draft_text

def verify_sequence_parallel(llm, tokenizer, history_text, draft_ids, tau):
    draft_text = tokenizer.decode(draft_ids)
    full_text = history_text + draft_text
    
    params = SamplingParams(max_tokens=1, prompt_logprobs=20, temperature=0.1) 
    outputs = llm.generate(prompts=[full_text], sampling_params=params, use_tqdm=False)
    
    all_logprobs = outputs[0].prompt_logprobs
    history_ids = tokenizer.encode(history_text)
    start_index = len(history_ids)
    
    valid_tokens = []
    correction_token = None
    stop_reason = "Completed"

    for i, token_id in enumerate(draft_ids):
        current_idx = start_index + i
        if current_idx >= len(all_logprobs): break
        
        logprobs_entry = all_logprobs[current_idx]
        if not logprobs_entry: continue 

        cloud_prob = np.exp(logprobs_entry[token_id].logprob) if token_id in logprobs_entry else 1e-9
        local_best_id = max(logprobs_entry, key=lambda k: logprobs_entry[k].logprob)
        local_best_prob = np.exp(logprobs_entry[local_best_id].logprob)
        
        ratio = cloud_prob / local_best_prob
        
        if ratio < tau:
            correction_token = local_best_id
            stop_reason = f"Ratio {ratio:.3f} < {tau}"
            break
        else:
            valid_tokens.append(token_id)

    return valid_tokens, correction_token, stop_reason

def run_p3(cloud_llm, local_llm, data_sample, k_draft=5, tau=0.05, max_steps=20, num_contexts=3, llm_type="qwen"):
    tokenizer = local_llm.get_tokenizer()
    public_prompt, private_prompt, profile_text = create_dual_prompts(data_sample, tokenizer, num_contexts=num_contexts, llm_type=llm_type)
    
    # Identify PII in the local profile to filter it from Cloud history later
    pii_list = get_pii_entities(profile_text)
    
    execution_logs = []
    cloud_history = public_prompt
    local_history = private_prompt
    final_generated_text = ""

    for step in range(max_steps):
        step_log = {"step": step + 1, "action": None, "details": {}}
        
        draft_ids, draft_text = get_k_tokens_cloud(cloud_llm, cloud_history, k_draft)
        step_log["cloud_draft"] = draft_text

        valid_ids, correction_id, reason = verify_sequence_parallel(
            local_llm, tokenizer, local_history, draft_ids, tau
        )
        
        accepted_text = tokenizer.decode(valid_ids)
        
        if correction_id is not None:
            correction_text = tokenizer.decode([correction_id])
            step_log["action"] = "INTERVENTION"
            step_log["details"] = {"reason": reason, "correction": correction_text}
            block_text = accepted_text + correction_text
        else:
            step_log["action"] = "ACCEPTED"
            block_text = accepted_text

        step_log["final_block"] = block_text
        execution_logs.append(step_log)

        final_generated_text += block_text
        
        local_history += block_text
        
        cloud_history += filter_pii_from_text(block_text, pii_list)
        
        if "<|endoftext|>" in block_text or tokenizer.eos_token in block_text:
            final_generated_text = final_generated_text.replace("<|endoftext|>", "").replace(tokenizer.eos_token, "")
            break
        if "<end_of_turn>" in block_text:
            final_generated_text = final_generated_text.replace("<end_of_turn>", "")
            break
            
    return {"output": final_generated_text, "logs": execution_logs}


parser = argparse.ArgumentParser()
parser.add_argument("--questions_address", type=str, required=True)
parser.add_argument("--output_address", type=str, required=True)
parser.add_argument("--cloud_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--local_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
parser.add_argument("--cache_dir", type=str, default=None)
parser.add_argument("--k_draft", type=int, default=10)
parser.add_argument("--tau", type=float, default=0.05)
parser.add_argument("--max_steps", type=int, default=10000)
parser.add_argument("--num_contexts", type=int, default=3)
parser.add_argument("--max_gen_tokens_cloud", type=int, default=8192)
parser.add_argument("--max_gen_tokens_local", type=int, default=8192)
parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--llm_type", type=str, default="qwen")

if __name__ == "__main__":
    args = parser.parse_args()

    SAVE_EVERY_M_STEPS = args.batch_size
    CACHE_DIR = args.cache_dir
    
    cloud_llm = LLM(model=args.cloud_model, gpu_memory_utilization=0.45, enforce_eager=True, download_dir=CACHE_DIR, max_model_len=args.max_gen_tokens_cloud, tensor_parallel_size=args.num_gpus, swap_space=40, enable_chunked_prefill=True)
    local_llm = LLM(model=args.local_model, gpu_memory_utilization=0.15, enforce_eager=True, download_dir=CACHE_DIR, max_model_len=args.max_gen_tokens_local, tensor_parallel_size=args.num_gpus, swap_space=40, enable_chunked_prefill=True)

    with open(args.questions_address, "r") as f:
        questions = json.load(f)
    
    output_dir = os.path.dirname(args.output_address)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir if output_dir else ".", "temp_results")
    os.makedirs(temp_dir, exist_ok=True)

    processed_ids = set()
    existing_results = []
    temp_files = glob.glob(os.path.join(temp_dir, "batch_*.json"))
    for tf in temp_files:
        with open(tf, "r") as f:
            batch_data = json.load(f)
            existing_results.extend(batch_data)
            for item in batch_data: processed_ids.add(item.get("id"))

    current_batch = []
    batch_counter = len(temp_files)

    for i, data in enumerate(tqdm.tqdm(questions)):
        sample_id = data.get("id")
        if sample_id in processed_ids: continue

        try:
            result = run_p3(cloud_llm, local_llm, data, args.k_draft, args.tau, args.max_steps, args.num_contexts, args.llm_type)
        except Exception as e:
            print(f"Error on sample {sample_id}: {e}")
            result = {"output": "", "logs": [], "error": str(e)}

        result["question"] = data["question"]
        result['id'] = sample_id
        current_batch.append(result)
        existing_results.append(result)

        if len(current_batch) >= SAVE_EVERY_M_STEPS:
            with open(os.path.join(temp_dir, f"batch_{batch_counter}.json"), "w") as f:
                json.dump(current_batch, f, indent=4)
            current_batch, batch_counter = [], batch_counter + 1

    if current_batch:
        with open(os.path.join(temp_dir, f"batch_{batch_counter}.json"), "w") as f:
            json.dump(current_batch, f, indent=4)

    with open(args.output_address, "w") as f:
        outputs = {x['id']: x for x in existing_results}
        json.dump(outputs, f, indent=4)