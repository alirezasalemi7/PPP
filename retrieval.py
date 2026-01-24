import torch
from transformers import AutoModel, AutoTokenizer
import json
import tqdm
import argparse

def batchify(lst, batch_size):
    return [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def retrieve_top_k_with_contriver(contriver, tokenizer, corpus, profile, query, k, batch_size):
    query_tokens = tokenizer([query], padding=True, truncation=True, return_tensors='pt').to("cuda:0")
    output_query = contriver(**query_tokens)
    output_query = mean_pooling(output_query.last_hidden_state, query_tokens['attention_mask'])
    scores = []
    batched_corpus = batchify(corpus, batch_size)
    for batch in batched_corpus:
        tokens_batch = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to("cuda:0")
        outputs_batch = contriver(**tokens_batch)
        outputs_batch = mean_pooling(outputs_batch.last_hidden_state, tokens_batch['attention_mask'])
        temp_scores = output_query.squeeze() @ outputs_batch.T
        scores.extend(temp_scores.tolist())
    topk_values, topk_indices = torch.topk(torch.tensor(scores), k)
    return [profile[m] for m in topk_indices.tolist()]

parser = argparse.ArgumentParser()
parser.add_argument("--input_dataset_addr", type=str, required=True)
parser.add_argument("--output_dataset_addr", type=str, required=True)
parser.add_argument("--model_name", type=str, default='facebook/contriever-msmarco')
parser.add_argument("--batch_size", type=int, default=64)

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.input_dataset_addr, "r") as file:
        dataset = json.load(file)
    rank_dict = dict()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    contriver = AutoModel.from_pretrained(args.model_name).to("cuda:0")
    contriver.eval()

    dataset_new = []
    for data in tqdm.tqdm(dataset):
        profile = data['profile']
        corpus = [x['text'] for x in profile]
        query = data['question']
        ids = [x['id'] for x in profile]
       
        ranked_profile = retrieve_top_k_with_contriver(contriver, tokenizer, corpus, profile, query, len(profile), args.batch_size)
        ranked_profile = [{'id': ids[i], 'text': ranked_profile[i]['text']} for i in range(len(ranked_profile))]        
        data['profile'] = ranked_profile
        dataset_new.append(data)

    
    with open(args.output_dataset_addr, "w") as file:
        json.dump(dataset_new, file, indent=4)