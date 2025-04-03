from openai import OpenAI
import yaml
import argparse
import os
import json

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
SECRETS_PATH = os.path.join(ROOT_DIR, 'secrets.yaml')

CONFIG = {
    "logprobs": True,
    "top_logprobs": 20,
    "temperature": 1,
    "max_completion_tokens": 1
}


def annotate_prompts(prompts_df, config):
    batch_file_name = os.path.basename(config['output']).removesuffix('.csv')
    if config['start_row'] is not None:
        batch_file_name += f"-{config['start_row']}"

    if config['end_row'] is not None:
        batch_file_name += f"-{config['end_row']}"
    else:
        batch_file_name += "-end"

    batch_file_name = batch_file_name + '.jsonl'
    batch_path = os.path.join(ROOT_DIR, 'openai_batches', batch_file_name)

    tasks = []
    for _, row in prompts_df.iterrows():
        custom_id = f"request-r{row['row_id']}-p{row['principle_id']}-o{row['order']}"
        task = {
            'custom_id': custom_id,
            'method': 'POST',
            "url": "/v1/chat/completions",
            'body': {
                "messages": row['prompt'],
                "model": config['model_id'].split('/')[-1]
            } | CONFIG
        }
        tasks.append(task)

    if len(tasks) == 0:
        print("No tasks to process.")
        return

    with open(batch_path, 'w') as f:
        for task in tasks:
            f.write(json.dumps(task) + '\n')

    print(f"Batch file written to {batch_path} with {len(tasks)} tasks.")

    confirm = input("Submit batch to OpenAI? (y/n): ")
    if confirm == 'y' or confirm == 'Y':
        submit_batch(batch_path)
    else:
        print("Batch not submitted.")


def submit_batch(batch_input_file):
    if not os.path.exists(batch_input_file):
        batch_input_file = os.path.join(ROOT_DIR, 'openai_batches', batch_input_file)
    assert os.path.exists(batch_input_file)
    assert batch_input_file.endswith('.jsonl')

    secrets = yaml.safe_load(open(SECRETS_PATH))
    client = OpenAI(api_key=secrets["OPENAI_ME"])
    batch_input_file = client.files.create(
        file=open(batch_input_file, 'rb'),
        purpose="batch"
    )
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "test_run"
        }
    )
    print(batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_input_file", type=str)
    submit_batch(parser.parse_args().batch_input_file)
