
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

    imdb_alignment_train(config)