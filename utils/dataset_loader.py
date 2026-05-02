import datasets


def load_imdb_pref_dataset(split="train", negative_sentiment=False, filter_rewards=True):
    # reward_model = AutoModelForSequenceClassification.from_pretrained("imdb_reward_model", device_map='cuda')
    # reward_tokenizer = AutoTokenizer.from_pretrained("imdb_reward_model")
    def remove_prompt(example):
        prompt_len = len(example['prompt'])

        whitespace = " " if (example['chosen'][0] != ' ' and example['prompt'][-1]!=' ') else ""
        example['chosen'] = whitespace+example['chosen'][prompt_len:] 

        whitespace = " " if (example['rejected'][0] != ' ' and example['prompt'][-1]!=' ') else ""
        example['rejected'] = whitespace+example['rejected'][prompt_len:] 
        return example
    
    def swap_prefs(example):
        example['chosen'], example['rejected'] = example['rejected'], example['chosen']
        return example
    
    processed_ds = datasets.load_dataset("yuasosnin/imdb-dpo", split=split)
    
    if filter_rewards:
        processed_ds = processed_ds.filter(lambda x: x["chosen_reward"] > 0 > x["rejected_reward"])
        
    processed_ds = processed_ds.map(remove_prompt)
    processed_ds = processed_ds.remove_columns(['chosen_reward', 'rejected_reward'])
    if negative_sentiment:
        processed_ds = processed_ds.map(swap_prefs)

    return processed_ds.shuffle(seed=42)

def load_imdb_sft_dataset():
    train_dataset = datasets.load_dataset("imdb", split='train')
    train_dataset = train_dataset.shuffle(seed=42).select(range(15000))
    train_dataset = train_dataset.remove_columns('label')
    train_dataset = train_dataset.shuffle(seed=42)

    eval_dataset = datasets.load_dataset("imdb", split='test')
    eval_dataset = eval_dataset.shuffle(seed=42).select(range(500))
    eval_dataset = eval_dataset.remove_columns('label')
    eval_dataset = eval_dataset.shuffle(seed=42)
    return train_dataset, eval_dataset