import pandas as pd
import litellm
from litellm import acompletion
import os
import numpy as np
from tqdm import tqdm
import asyncio

from dotenv import load_dotenv
load_dotenv('secrets.env')


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

CONFIG = {
    "logprobs": True,
    "top_logprobs": 20,
    "temperature": 1,
    "max_completion_tokens": 1
}

PARALLEL_SIZE = 20
MAX_RETRIES = 10
DEBUG = False


async def post_request(**kwargs):
    attempts = 0
    delay = 2  # Initial delay in seconds
    response = None
    while attempts < MAX_RETRIES:
        try:
            response = await acompletion(**kwargs)
            if DEBUG:
                print(kwargs['messages'])
                print(response)
            return response
        except asyncio.TimeoutError:
            print("The request has timed out.")
        except Exception as e:
            attempts += 1
            print("Request failed: ", str(e))
            if attempts < MAX_RETRIES:
                await asyncio.sleep(delay)  # Wait before retrying
                delay *= 2  # Exponential backoff
    if response is None:
        raise Exception("All retry attempts failed.")


async def post_many_requests(prompts_df, config, existing_result_df=None):
    model = config['model_id']

    requested_tokens = ["A", "B"] + (["NA"] if config['include_na'] else [])

    full_score_df = existing_result_df
    tasks = []
    for _, row in prompts_df.iterrows():
        messages = row['prompt']
        tasks.append(post_request(model=model, messages=messages, **CONFIG))
    with tqdm(total=len(tasks), desc="Processing") as pbar:
        for i in range(0, len(tasks), PARALLEL_SIZE):
            end = i + PARALLEL_SIZE if i + PARALLEL_SIZE < len(tasks) else len(tasks)
            tasks_parallel = tasks[i:end]
            responses = await asyncio.gather(*tasks_parallel)
            pbar.update(len(tasks_parallel))

            token_probs = {f'prob_{t}': [] for t in requested_tokens}
            for response in responses:

                for requested_token in requested_tokens:
                    all_logprobs = response['choices'][0]['logprobs']['content'][0]['top_logprobs']
                    requested_prob = 0.
                    for logprob in all_logprobs:
                        if logprob['token'] == requested_token:
                            requested_prob = np.exp(logprob['logprob'])
                            break
                    token_probs[f'prob_{requested_token}'].append(requested_prob)
            scores_df = pd.DataFrame(token_probs | {
                'order': prompts_df['order'].iloc[i:end],
                'row_id': prompts_df['row_id'].iloc[i:end],
            })
            if full_score_df is None:
                scores_df.to_csv(config['output'], index=False)
                full_score_df = scores_df
            else:
                scores_df.to_csv(config['output'], mode='a', index=False, header=False)
                full_score_df = pd.concat([full_score_df, scores_df], ignore_index=True)
    return full_score_df


def annotate_prompts(prompts_df, config, existing_result_df=None):
    print(f"Processing {len(prompts_df)} prompts with model {config['model_id']}.")
    # litellm._turn_on_debug()
    full_score_df = asyncio.run(post_many_requests(prompts_df, config, existing_result_df))
    return full_score_df
