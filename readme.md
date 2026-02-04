# Codes and data for the paper "[Improving User Privacy in Personalized Generation: Client-Side Retrieval-Augmented Modification of Server-Side Generated Speculations](https://arxiv.org/abs/2601.17569)"

Personalization is crucial for aligning Large Language Model (LLM) outputs with individual user preferences and background knowledge. State-of-the-art solutions are based on retrieval augmentation where relevant context from user profile is retrieved for LLM consumption. These methods deal with a trade-off between exposing retrieved private data to cloud providers and relying on less capable local models. We introduce $P^3$, an interactive framework for high-quality personalization without revealing private profiles to server-side LLMs. In $P^3$, a large server-side model generates a sequence of $k$ draft tokens based solely on the user query, while a small client-side model, with retrieval access to the user’s private profile, evaluates and modifies these drafts to better reflect user preferences. This process repeats until an end token is generated. Experiments on LaMP-QA, a recent benchmark consisting of three personalized question answering datasets, show that $P^3$ consistently outperforms both non-personalized server-side and personalized client-side baselines, achieving statistically significant improvements of 7.4% to 9% on average. Importantly, $P^3$ recovers 90.3% to 95.7% of the utility of a “leaky” upper-bound scenario in which the full profile is exposed to the large server-side model. Privacy analyses, including linkability and attribute inference attacks, indicate that $P^3$ preserves the privacy of a non-personalized server-side model, introducing only marginal additional leakage (1.5%–3.5%) compared to submitting a query without any personal context. Additionally, the framework is efficient for edge deployment, with the client-side model generating only 9.2% of the total tokens. These results demonstrate that $P^3$ provides a practical, effective solution for personalized generation with improved privacy.


## Requirements

```bash
pip install -r requirements.txt
```

## Preparing Dataset

### Downloading the Data

First, you need to download the [LaMP-QA](https://github.com/LaMP-Benchmark/LaMP-QA?tab=readme-ov-file#downloading-the-dataset) dataset. For this purpose, you can use the following code:

```bash
python download.py \
    --dataset_save_directory /*address to the download directory*/ \
    --cache_dir /*address to the cache directory ([optional], default is ./cache)*/
```

### Retrieval from User Profile

Next, in order to prepare the RAG personalization on the client-side, we need to rank the dataset for each query and prepare it in a format that is usable for $P^3$. To do this, you can use the following code:

```bash
python retrieval.py \
    --input_dataset_addr /*address to the dataset file*/ \
    --output_dataset_addr /*address to where the dataset with sorted profile for each user should be saved*/ \
    --model_name "facebook/contriever-msmarco" \
    --batch_size 16
```

After this step, the dataset is ready to be used for $P^3$.

## Running $P^3$

In order to use $P^3$, run the following code:

```bash
python PPP.py \
    --questions_address /*address to the prepared dataset*/ \
    --output_address /*address to the output directory*/ \
    --cloud_model /*cloud model, must be larger than local*/ \
    --local_model /*local model, must be smaller than cloud*/ \
    --k_draft /*number of draft tokens by cloud model*/ \
    --tau /*rejection threshold*/ \
    --num_contexts /*number of retrieved document from user profile*/ \
    --max_gen_tokens_cloud /*maximum context len by cloud model*/ \
    --max_gen_tokens_local /*maximum context len by local model*/ \
    --num_gpus /*number of available GPUs*/ \
    --batch_size /*size of batch for generation and saving*/ \
    --llm_type /*qwen or gemma*/
```

This code will create a directory with the following structure:

```
directory\
    temp_results\ \*a directory used for storing intermediate results and helps with resuming generation in case of interrupt*\
    results.json  \*the response to the queries*\
```

The result file has the following structure:
```json
{
    "/*the question id*/" : {
        "question": "/*the given question*/",
        "id": "/*the question id*/",
        "final_response": "/*response to the question by P3*/",
        "logs": [
            {
                "step": 1 /*step number*/,
                "action": "/*INTERVENTION if correction happened and ACCEPTED if all tokens are accepted*/",
                "details": {
                    "reason": "/*reason for rejection based on ratios*/",
                    "correction": "/*the token that was corrected*/"
                },
                "cloud_draft": "/*draft tokens from the server-side model*/",
                "final_block": "/*final text including accepted tokens and corrected token in this step*/"
            },
            ...
        ]
    },
    ...
}
```

## Evaluation

Please use the evaluation script provided by the [LaMP-QA](https://github.com/LaMP-Benchmark/LaMP-QA?tab=readme-ov-file#evaluating-the-generated-responses) benchmark to evaluate generated responses.

## Reference

[Improving User Privacy in Personalized Generation: Client-Side Retrieval-Augmented Modification of Server-Side Generated Speculations](https://arxiv.org/abs/2601.17569)

```
@misc{salemi2026improvinguserprivacypersonalized,
      title={Improving User Privacy in Personalized Generation: Client-Side Retrieval-Augmented Modification of Server-Side Generated Speculations}, 
      author={Alireza Salemi and Hamed Zamani},
      year={2026},
      eprint={2601.17569},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.17569}, 
}
```

## Acknowledgment

This work was supported in part by the Center for Intelligent Information Retrieval, in part by NSF grant #2143434, in part by the Office of Naval Research contract #N000142412612, and with support from by Google.org. Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect those of the sponsor.

