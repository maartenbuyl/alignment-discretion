from datasets import load_dataset
import pandas as pd
import numpy as np


def load_huggingface(dataset_name=None, split=None, principles_name=None, **_kwargs):
    if dataset_name is None:
        raise ValueError("dataset_name must be specified!")

    # Load raw data from HuggingFace
    dataset = load_dataset(dataset_name)

    if dataset_name == "Anthropic/hh-rlhf":
        data = process_hh_rlhf(dataset, split=split)

    elif dataset_name == "PKU-Alignment/PKU-SafeRLHF":
        data = process_pku(dataset, split=split)
    elif dataset_name == "PKU-Alignment/PKU-SafeRLHF-single-dimension":
        data = process_pku_single(dataset, split=split)
    else:
        raise ValueError(f"Dataset name {dataset_name} unknown!")
    
    if principles_name is None:
        return data
    
    with open(principles_name, 'r') as f:
        principles = [line.strip() for line in f.readlines()]
    principles = pd.Series(principles)
    return data, principles


def prompt(text):
    last_assistant = text.rfind("Assistant:")
    return text[:last_assistant].strip()


def answer(text):
    last_assistant = text.rfind("Assistant:")
    return text[last_assistant:].strip()


def process_hh_rlhf(dataset, split=None):
    df = pd.DataFrame({
        "prompt": [prompt(text) for text in dataset[split]['chosen']],
        "chosen": [answer(text) for text in dataset[split]['chosen']],
        "rejected": [answer(text) for text in dataset[split]['rejected']],
        "a_pref": np.ones(len(dataset[split])),
    })
    df["row_id"] = df.index.values
    return df


def process_pku_single(dataset, split=None):
    df = pd.DataFrame(dataset[split])
    df = pd.DataFrame({
        'row_id': df.index,
        'prompt': df['prompt'],
        'chosen': df['response_1'],
        'rejected': df['response_0'],
        'a_pref_single': df['better_response_id'] * 2 - 1,
    })
    return df


def process_pku(dataset, split=None):
    df = pd.DataFrame(dataset[split])
    df = pd.DataFrame({
        'row_id': df.index,
        'prompt': df['prompt'],
        'chosen': df['response_1'],
        'rejected': df['response_0'],
        'a_pref_better': df['better_response_id'] * 2 - 1,
        'a_pref_safer': df['safer_response_id'] * 2 - 1,
        # 'a_pref_less_severe': np.sign(df['response_0_severity_level'] - df['response_1_severity_level']),
    })
    return df
