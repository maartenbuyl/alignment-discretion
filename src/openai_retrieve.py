from openai import OpenAI
import yaml
import argparse
import os
import pandas as pd
import numpy as np
import io

from dotenv import load_dotenv
load_dotenv('secrets.env')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# In $ / million tokens. As of 2025-01-11.
PRICES = {
    'gpt-4o-2024-11-20': {
        'input': 1.25,
        'output': 5.
    },
    'gpt-4o-2024-08-06': {
        'input': 1.25,
        'output': 5.
    },
    # 'gpt-4o': {
    #     'input': 2.5,
    #     'output': 10.
    # },
    # 'gpt-4o-mini': {
    #     'input': 0.15,
    #     'output': 0.6
    # }
}


def retrieve_batch(batch_id=None, include_na=False, output=None):
    client = OpenAI(api_key=os.environ["OPENAI_ME"])

    status = client.batches.retrieve(batch_id)
    print(f"Batch status: {status}")

    output_file_id = status.output_file_id
    responses = client.files.content(output_file_id).text
    print(f"Batch responses: {responses}")

    df = pd.read_json(io.StringIO(responses), lines=True)
    df[['row_id', 'principle_id', 'order']] = df['custom_id'].str.extract(r'request-r(\d+)-p(\d+)-o(\d+)')
    df['row_id'] = df['row_id'].astype(int)
    df['principle_id'] = df['principle_id'].astype(int)
    df['order'] = df['order'].astype(int)

    requested_tokens = ["A", "B"] + (["NA"] if include_na else [])
    valid_responses = df['response'].apply(
        lambda row: len(row['body']['choices'][0]['logprobs']['content']) > 0
    )
    if not valid_responses.all():
        print(f"Warning: {len(valid_responses) - valid_responses.sum()} responses are invalid. Dropping them...")
        df = df[valid_responses]
    top_log_probs = df['response'].apply(
        lambda row: row['body']['choices'][0]['logprobs']['content'][0]['top_logprobs']
    )
    for token in requested_tokens:
        df[f'prob_{token}'] = top_log_probs.apply(
            lambda probs: [np.exp(prob['logprob']) for prob in probs if prob['token'] == token]
        )
        df[f'prob_{token}'] = df[f'prob_{token}'].apply(lambda probs: probs[0] if len(probs) > 0 else 0.)
    model_id = df['response'].apply(lambda row: row['body']['model']).iloc[0]
    df['cost'] = df['response'].apply(lambda row: estimate_cost(row['body']['usage'], model_id))

    df = df.drop(columns=['custom_id', 'response', 'id', 'error'])

    print(f"Batch is estimated to have cost {df['cost'].sum()} USD.")

    if output is None:
        raw_output_path = os.path.join(ROOT_DIR, 'principle_specific_preferences', f'{batch_id}.csv')
    elif not os.path.exists(output):
        raw_output_path = os.path.join(ROOT_DIR, 'principle_specific_preferences', output)
    else:
        raw_output_path = output

    if not raw_output_path.endswith('.csv'):
        raise ValueError("Output file must be a CSV file.")
    elif os.path.exists(raw_output_path):
        existing_df = pd.read_csv(raw_output_path)
        if not existing_df.columns.equals(df.columns):
            raise ValueError("Existing output file has different columns.")
        existing_raveled = np.ravel_multi_index(tuple(existing_df[['row_id', 'principle_id', 'order']].T.values),
                                                dims=tuple(existing_df[['row_id', 'principle_id', 'order']].max() + 1))
        new_raveled = np.ravel_multi_index(tuple(df[['row_id', 'principle_id', 'order']].T.values),
                                           dims=tuple(df[['row_id', 'principle_id', 'order']].max() + 1))
        if np.intersect1d(existing_raveled, new_raveled).size > 0:
            raise ValueError(f"New output file has {np.intersect1d(existing_raveled, new_raveled).size} "
                             f"overlapping indices with existing output file.")

    if 'generic' in raw_output_path:
        df = df.drop(columns=['principle_id'])
        raw_output_path = raw_output_path.replace('principle_specific_preferences', 'llm_preferences')

    if os.path.exists(raw_output_path):
        df.to_csv(raw_output_path, mode='a', header=False, index=False)
    else:
        print(f"Writing new output file to {raw_output_path}")
        df.to_csv(raw_output_path, index=False)


def estimate_cost(usage_dict, model_id):
    model_id = model_id.split('/')[-1]
    input_tokens = usage_dict['prompt_tokens']
    output_tokens = usage_dict['completion_tokens']
    return PRICES[model_id]['input'] * input_tokens / 1e6 + PRICES[model_id]['output'] * output_tokens / 1e6


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_id", type=str)
    parser.add_argument("--include_na", action='store_true')
    parser.add_argument("--output", type=str, default=None, help="Path to store output file or append to if it exists")
    retrieve_batch(**vars(parser.parse_args()))
