
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import datasets
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from trl import DPOConfig
from utils.patched_dpo_trainer import CustomDPOTrainer
from utils.dataset_loader import load_imdb_pref_dataset 
import argparse
import json
import re


def imdb_alignment_train(cfg):
    model_name = cfg["model_name"]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = load_imdb_pref_dataset(split="train", negative_sentiment=cfg['negative_sentiment'], filter_rewards=cfg['filter_rewards'])
    eval_dataset =  load_imdb_pref_dataset(split="test", negative_sentiment=cfg['negative_sentiment'], filter_rewards=cfg['filter_rewards'])

    dpo_trainer = CustomDPOTrainer(
        model=model,
        ref_model=None,
        processing_class=tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        args = DPOConfig(
            **cfg['TrainerConfig']
        )
    )
    dpo_trainer.train()
    dpo_trainer.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alignment CLI")
    parser.add_argument("--cfg", type=str, help='Configuration file.')
    args = parser.parse_args()

    with open(f'./configs/trainers/{args.cfg}.json', 'r', encoding='utf-8') as file:
        config = json.load(file)

    beta = 0.25
    alphas = [0.1, 0.4, 0.5, 0.6, 0.9, 0.25, 0.75]
    for model_name in [
        "./models/LiquidAI/LFM2.5-350M-Base-sft-imdb-dpo-sentiment-b0.25",
        "./models/LiquidAI/LFM2.5-350M-Base-sft-imdb-ipo-sentiment-b0.25",
        "./models/LiquidAI/LFM2.5-350M-Base-sft-imdb-aspo-sentiment-b0.25-a0.1",
        "./models/LiquidAI/LFM2.5-350M-Base-sft-imdb-aspo-sentiment-b0.25-a0.4",
        "./models/LiquidAI/LFM2.5-350M-Base-sft-imdb-aspo-sentiment-b0.25-a0.5",
        "./models/LiquidAI/LFM2.5-350M-Base-sft-imdb-aspo-sentiment-b0.25-a0.6",
        "./models/LiquidAI/LFM2.5-350M-Base-sft-imdb-aspo-sentiment-b0.25-a0.9",
        "./models/LiquidAI/LFM2.5-350M-Base-sft-imdb-aspo-sentiment-b0.25-a0.25",
        "./models/LiquidAI/LFM2.5-350M-Base-sft-imdb-aspo-sentiment-b0.25-a0.75",
    ]:
        match = re.search(r'-b([\d.]+)', model_name)
        if match:
            beta = (match.group(1))
        config['TrainerConfig']['beta'] = float(beta)        
        config['TrainerConfig']["output_dir"] = model_name
        if "dpo" in model_name:
            config['TrainerConfig']['loss_type'] = "sigmoid"
        elif "ipo" in model_name :
            config['TrainerConfig']['loss_type'] = "ipo"
        else:
            config['TrainerConfig']['loss_type'] = "aspo"
            match = re.search(r'-a([\d.]+)', model_name)
            if match:
                alpha = (match.group(1))
            config['TrainerConfig']['label_smoothing'] = float(alpha)
        imdb_alignment_train(config)
