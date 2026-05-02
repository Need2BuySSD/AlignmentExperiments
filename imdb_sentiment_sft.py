from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import argparse
import json
from utils.dataset_loader import load_imdb_sft_dataset

def imdb_sft_train(cfg):
    model_name = cfg['model_name']

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset, eval_dataset = load_imdb_sft_dataset()

    sft_config = SFTConfig(
        **cfg["TrainerConfig"]
    )

    sft_trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    sft_trainer.train()
    sft_trainer.save_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alignment CLI")
    parser.add_argument("--cfg", type=str, help='Configuration file.')
    args = parser.parse_args()

    with open(f'./configs/trainers/{args.cfg}.json', 'r', encoding='utf-8') as file:
        config = json.load(file)

    imdb_sft_train(config)