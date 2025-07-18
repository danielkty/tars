"""Script for curating TARS RL train and test data adapted from DeepScaleR."""

import argparse
import os
from typing import Dict, List, Optional, Any
from datasets import load_dataset, concatenate_datasets
import random
import json

from strong_reject.load_datasets import load_strongreject_small

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs

def make_map_fn(split: str):
    if split == 'wildjailbreak':
        def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
            vanilla_question = example['vanilla']
            adversarial_question = example['adversarial']
            question = vanilla_question if adversarial_question == "" else adversarial_question
            completion = example['completion']

            data = {
                "data_source": "",
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "wildjailbreak",
                "reward_model": {
                    "style": "model",
                    "ground_truth": completion
                },
                "extra_info": {
                    'split': "train",
                    'index': idx
                }
            }
            return data
    elif split == 'default':
        def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
            question = example['prompt']
            data_type= example['datatype']

            data = {
                "data_source": "",
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": data_type,
                "reward_model": {
                    "style": "model",
                    "ground_truth": ""
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data
        
    elif split == 'test':
        def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
            question = example['forbidden_prompt']
            data_source = example['source']
            data_type= example['category']

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": data_type,
                "reward_model": {
                    "style": "model",
                    "ground_truth": ""
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data
    else:
        raise ValueError(f"Invalid split name: {split}")
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Curate dataset for TARS RL training and testing')
    parser.add_argument('--local_dir', default=os.path.expanduser('~/tars/data'), help='Local directory to save processed datasets')
    parser.add_argument('--train_dataset', type=str, default='allenai/wildjailbreak', help='Hugging Face dataset name for training')
    parser.add_argument('--test_dataset', type=str, default='strongreject', help='Hugging Face dataset name for testing')
    args = parser.parse_args()
    random.seed(0)
    prompt_length = 1000

    local_dir = args.local_dir
    
    # Make local directory if it doesn't exist
    makedirs(local_dir)

    safety_train_data: List[Dict[str, Any]] = []
 
    # Initialize datasets
    train_dataset = load_dataset(args.train_dataset, 'train', delimiter="\t", keep_default_na=False)
    train_dataset = train_dataset["train"].filter(lambda x: x["data_type"] in ["vanilla_harmful", "adversarial_harmful"])
    train_dataset = list(train_dataset)
    train_dataset = [prompt for prompt in train_dataset if len(prompt) < prompt_length]
    random.shuffle(train_dataset)
    train_dataset = train_dataset[400:1200]
    process_fn = make_map_fn('wildjailbreak')

    for idx, example in enumerate(train_dataset):
        processed_example = process_fn(example, idx)
        if processed_example is not None:
            safety_train_data.append(processed_example)

    adversarial_paths = [
        f'rainbow_archives/archive_1.json',
        f'rainbow_archives/archive_2.json',
        f'rainbow_archives/archive_3.json',
        f'rainbow_archives/archive_4.json',  
        f'rainbow_archives/archive_5.json',
        f'rainbow_archives/archive_6.json',
        f'rainbow_archives/archive_7.json',
    ]

    adversarial_prompts = []
    for adversarial_path in adversarial_paths:
        with open(adversarial_path, 'r') as f:
            adversarial_data = json.load(f)
        adversarial_prompts.extend([entry["prompt"] for row in adversarial_data for entry in row])
        
    adversarial_prompts = [prompt for prompt in adversarial_prompts if len(prompt) < prompt_length]
    adversarial_prompts = [{'prompt': prompt, 'datatype': 'rainbow'} for prompt in adversarial_prompts]
    adversarial_prompts = random.sample(adversarial_prompts, 800)

    process_fn = make_map_fn('default')

    for idx, example in enumerate(adversarial_prompts):
        processed_example = process_fn(example, idx)
        if processed_example is not None:
            safety_train_data.append(processed_example)

    train_dataset = load_dataset('nvidia/Aegis-AI-Content-Safety-Dataset-2.0')
    train_dataset = train_dataset["train"].filter(lambda x: x["prompt_label"] in ["unsafe"] and x["prompt"] != "REDACTED")
    aegis_prompts = train_dataset['prompt']
    aegis_prompts = [prompt for prompt in aegis_prompts if len(prompt) < prompt_length]
    random.shuffle(aegis_prompts)
    aegis_prompts = aegis_prompts[300:700]
    aegis_prompts = [{'prompt': prompt, 'datatype': 'aegis'} for prompt in aegis_prompts]

    for idx, example in enumerate(aegis_prompts):
        processed_example = process_fn(example, idx)
        if processed_example is not None:
            safety_train_data.append(processed_example)

    train_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    refusal_prompts = train_dataset['train_prefs']['prompt']
    refusal_prompts = [prompt for prompt in refusal_prompts if len(prompt) < prompt_length]
    refusal_prompts = random.sample(refusal_prompts, 2000)
    refusal_prompts = [{'prompt': prompt, 'datatype': 'refusal'} for prompt in refusal_prompts]

    refusal_train_data: List[Dict[str, Any]] = []

    for idx, example in enumerate(refusal_prompts):
        processed_example = process_fn(example, idx)
        if processed_example is not None:
            refusal_train_data.append(processed_example)

    ds = load_dataset("bench-llm/or-bench", "or-bench-hard-1k")
    nonrefusal_prompts_hard = ds["train"]["prompt"]
    ds = load_dataset("bench-llm/or-bench", "or-bench-80k")
    nonrefusal_prompts = ds["train"]["prompt"]
    nonrefusal_prompts = [prompt for prompt in nonrefusal_prompts if prompt not in nonrefusal_prompts_hard]
    nonrefusal_prompts = [prompt for prompt in nonrefusal_prompts if len(prompt) < prompt_length]
    nonrefusal_prompts = random.sample(nonrefusal_prompts, 2000)

    nonrefusal_prompts = [{'prompt': prompt, 'datatype': 'refusal'} for prompt in nonrefusal_prompts]

    nonrefusal_train_data: List[Dict[str, Any]] = []

    for idx, example in enumerate(nonrefusal_prompts):
        processed_example = process_fn(example, idx)
        if processed_example is not None:
            nonrefusal_train_data.append(processed_example)

    # Set lambda for harmful:harmless ratio
    safety_ratio = 0.5

    safety_train_data = random.sample(safety_train_data, int(len(safety_train_data) * safety_ratio))
    refusal_train_data = random.sample(refusal_train_data, int(len(refusal_train_data) * (1 - safety_ratio)/2))
    nonrefusal_train_data = random.sample(nonrefusal_train_data, int(len(nonrefusal_train_data) * (1 - safety_ratio)/2))
    train_data = safety_train_data + refusal_train_data + nonrefusal_train_data
    print(len(train_data), len(safety_train_data), len(refusal_train_data), len(nonrefusal_train_data))

    test_dataset = load_strongreject_small()

    test_data: List[Dict[str, Any]] = []
    process_fn = make_map_fn('test')
    for idx, example in enumerate(test_dataset):
        processed_example = process_fn(example, idx)
        if processed_example is not None:
            test_data.append(processed_example)
    test_df = pd.DataFrame(test_data)
    test_df.to_parquet(os.path.join(local_dir, 'test.parquet'))
    print(f"Safety test data size:", len(test_data))

    # Save training dataset
    train_df = pd.DataFrame(train_data)

    train_df.to_parquet(os.path.join(local_dir, f'train_lambda_{safety_ratio}.parquet'))
    print("Safety train data size:", len(train_data))