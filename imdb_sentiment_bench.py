from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import re
import datasets
import argparse
import json
from time import time

# siebert/sentiment-roberta-large-english
# nlptown/bert-base-multilingual-uncased-sentiment

class SentAnalyzer:
    def __init__(self):
        self.sentiment_analysis_model = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

    @torch.no_grad()
    def __call__(self, texts):
        score = 0
        
        for x in self.sentiment_analysis_model(texts, ):
            score += int(x['label']=="NEGATIVE")
        return score


def benchmark_model(sentiment_pipe, model, ref_model, tokenizer, prompts, system_prompt=None, reply=None, T=1.0, chat_template=False, batch_size=32, n_respones=4):
    """
    DPO paper Sentiment generation 
    """
    model.eval()
    ref_model.eval()
    kl_values = []
    sentiment_scores = []
    n_batch_prompts = batch_size//n_respones
    n_samples = len(prompts) * n_respones
    with torch.no_grad():
        for i in range(0, len(prompts), n_batch_prompts):
            batch_prompts = prompts[i:i+n_batch_prompts]
            inputs, prefix_len = prepare_inputs(tokenizer, batch_prompts, system_prompt=system_prompt, reply=reply, chat_template=chat_template, n_responses=n_respones)
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=T,
                return_dict_in_generate=True,
                output_scores=True
            ).sequences

            if chat_template:
                decode_tokens = outputs[:, prefix_len:]
            else:
                decode_tokens = outputs
            texts = tokenizer.decode(decode_tokens, skip_special_tokens=True)

            sentiment_scores.append(sentiment_pipe(texts))
            kl_values.append(estimate_kl(tokenizer, model, ref_model, outputs, prefix_len))

    return np.sum(kl_values) / (n_samples), np.sum(sentiment_scores) / (n_samples)


def estimate_kl(tokenizer, model, ref_model, full_ids, prefix_len):
    """
    E_{y ~ pi_theta}[log(pi_theta(y|x) / pi_ref(y|x))]
    """
    attention_mask = (full_ids != tokenizer.pad_token_id).bool()

    logits = model(full_ids, attention_mask=attention_mask).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(
        log_probs[:, :-1, :],
        dim=-1,
        index=full_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)

    ref_logits = ref_model(full_ids, attention_mask=attention_mask).logits
    ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
    ref_token_log_probs = torch.gather(
        ref_log_probs[:, :-1, :],
        dim=-1,
        index=full_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)

    mask = attention_mask[:, 1:]
    mask[:, :prefix_len-1] = 0

    kl_value = ((token_log_probs - ref_token_log_probs) * mask.float()).sum(dim=-1).sum(dim=0).cpu().item()
    return kl_value

def prepare_inputs(tokenizer, prompts, system_prompt=None, reply=None, chat_template=False, n_responses=4,):
    if chat_template:
        raise NotImplementedError
        conversation = [] 
        if system_prompt is not None:
            conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": prompt})

        history_len = len((tokenizer.apply_chat_template(conversation, add_generation_prompt=True))['input_ids'])
        
        continue_reply = reply is not None
        if continue_reply:
            conversation.append({"role": "assistant", "content": reply})
            
        
        inputs = tokenizer.apply_chat_template([conversation]*n_responses, 
                                                add_generation_prompt=not continue_reply,
                                                continue_final_message=continue_reply,
                                                return_tensors="pt").to("cuda")
    else:
        inputs = tokenizer(prompts*n_responses, return_tensors="pt", padding=True, padding_side='left').to("cuda")
        prefix_len = inputs['input_ids'].shape[1]

    return inputs, prefix_len

def run_full_bench(config):
    ref_model_name = config['ref_model_name']
    model_names = config['model_names']
    batch_size = config['batch_size']
    
    results = {}
    tokenizer = AutoTokenizer.from_pretrained(ref_model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_name,
        device_map="cuda",
        torch_dtype="auto",
    )
    sentiment_analysis = SentAnalyzer()

    ds = datasets.load_dataset("yuasosnin/imdb-dpo", split='test').shuffle(seed=42)
    prompts = list(set(ds['prompt']))[:250]

    for model_name in model_names:
        checkpoint_list = [d for d in os.listdir(f'{model_name}') if d.startswith('checkpoint-')]
        checkpoint_list.sort(key=lambda x: int(re.search(r'\d+', x).group()))

        kl_list = []
        acc_list = []
        for model_checkpoint in checkpoint_list:
            aligned_model = AutoModelForCausalLM.from_pretrained(
                f"{model_name}/{model_checkpoint}",
                device_map="cuda",
                torch_dtype="auto",
            )
            kl, acc = benchmark_model(sentiment_analysis, aligned_model, ref_model, tokenizer, prompts=prompts, batch_size=batch_size)
            kl_list.append(kl)
            acc_list.append(acc)
        results[model_name] = {"kl" : kl_list, "acc": acc_list}
    
    return results





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alignment CLI")
    parser.add_argument("--cfg", type=str, help='Configuration file.')
    args = parser.parse_args()
    
    timestamp_int = int(time())

    with open(f'./configs/benchmarks/{args.cfg}.json', 'r', encoding='utf-8') as file:
        config = json.load(file)

    experiment_tag = config['experiment_tag']

    data = run_full_bench(config)
    with open(f'./results/{timestamp_int}_{experiment_tag}.json', 'w') as file:
        json.dump(data, file)