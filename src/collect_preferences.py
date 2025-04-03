import os
import torch
import pandas as pd
import argparse

from load_huggingface import load_huggingface
from prompt_template import form_chat_prompt, form_generic_chat_prompt
from openai_submit import annotate_prompts as annotate_prompts_batch_mode
from litellm_submit import annotate_prompts as annotate_prompts_litellm
from local_submit import annotate_prompts as annotate_prompts_local

# Constants
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DEFAULTS = {
    'dataset_name': None,
    # "dataset_name": "Anthropic/hh-rlhf",
    # "dataset_name": "PKU-Alignment/PKU-SafeRLHF",
    "principles_name": None,
    "model_name": "qwen1b-i",
    "start_row": None,
    "end_row": None,
    "split": None,
    "batch_size": 5,
    "torch_dtype": torch.float16,
    "devices": 'mps',
    "include_na": True,
    "include_reversed": False,
    "continue": False,
    "output": 'output.csv'
}
MODEL_INFO_MAP = {
    'qwen1b-i': {
        'model_id': 'Qwen/Qwen2.5-1.5B-Instruct'
    },
    'llama8b-i': {
        'model_id': "meta-llama/Llama-3.1-8B-Instruct"
    },
    'llama70b-i': {
        'model_id': "meta-llama/Llama-3.1-70B-Instruct"
    },
    '4o-mini': {
        'model_id': "openai/gpt-4o-mini"
    },
    '4o': {
        'model_id': "openai/gpt-4o"
    }
}


def main(config):
    # Configure devices (only really relevant when running locally)
    devices = config['devices']
    if devices is None:
        if torch.cuda.device_count() > 1:
            devices = ["cuda:0", "cuda:1"]
        elif torch.backends.mps.is_available():
            devices = 'mps'
        else:
            devices = "cpu"
    elif isinstance(devices, list) and len(devices) == 1:
        devices = devices[0]
    config['devices'] = devices

    # Map model shorthand to model name
    if config['model_name'] in MODEL_INFO_MAP:
        config |= MODEL_INFO_MAP[config['model_name']]
    else:
        config['model_id'] = config['model_name']

    if 'principles_name' is None:
        print("No principles_name provided, so collecting generic preferences.")
    collecting_generic_preferences = config['principles_name'] is None

    # Check/configure output path
    assert config['output'].endswith('.csv')
    if not config['output'].startswith('/'):
        output_dir = "principle_specific_preferences" if not collecting_generic_preferences else "llm_preferences"
        config['output'] = os.path.join(ROOT_DIR, output_dir, config['output'])
        os.makedirs(os.path.dirname(config['output']), exist_ok=True)

    # Load data and principles (if principle name provided)
    if config['principles_name'] is not None:
        if not os.path.exists(config['principles_name']):
            config['principles_name'] = os.path.join(ROOT_DIR, "principles", config['principles_name'])
        df, principles = load_huggingface(**config)
    else:
        assert collecting_generic_preferences
        df = load_huggingface(**config)
        principles = None

    # Collect subset of data 
    if config['start_row'] is not None:
        df = df[df['row_id'] >= config['start_row']]
    if config['end_row'] is not None:
        df = df[df['row_id'] < config['end_row']]

    prompts_df = compose_prompts(df, principles, config)

    # For composed prompts, check if we didn't already answer them
    existing_result_df = None
    if config['continue']:
        if not os.path.exists(config['output']):
            print(f"Script called with 'continue', but output file {config['output']} does not exist!\n"
                  f"Creating new file instead...")
        else:
            existing_result_df = pd.read_csv(config['output'])
            if config['start_row'] is not None:
                existing_result_df = existing_result_df[existing_result_df['row_id'] >= config['start_row']]
            cols_to_match = ['row_id', 'order'] + (['principle_id'] if 'principle_id' in existing_result_df else [])
            check_matches = (prompts_df.loc[existing_result_df.index, cols_to_match] ==
                             existing_result_df[cols_to_match])
            if not check_matches.all().all():
                raise ValueError("Existing results do not match the row_id, principle_id, and order "
                                 "expected for the prompts.")
            prompts_df = prompts_df.drop(existing_result_df.index)

    # Send prompts to models
    if config['model_id'].startswith("openai/"):
        annotate_prompts_batch_mode(prompts_df, config)
        return  # The code stops here, because we collect responses later with batch mode
    elif config['model_id'].startswith("deepseek/") or config['model_id'].startswith("anthropic/"):
        result_df = annotate_prompts_litellm(prompts_df, config, existing_result_df=existing_result_df)
    else:
        result_df = annotate_prompts_local(prompts_df, config, existing_result_df=existing_result_df)

    if collecting_generic_preferences:
        return  # no post-processing needed for generic preferences (i.e. not related to principles)

    # Manually pivot the results to have each column be a principle-specific preference
    for principle_id in result_df['principle_id'].unique():
        result_df[f'prob_A_p{principle_id}'] = result_df[result_df['principle_id'] == principle_id]['prob_A']
        result_df[f'prob_B_p{principle_id}'] = result_df[result_df['principle_id'] == principle_id]['prob_B']
        if config['include_na']:
            result_df[f'prob_NA_p{principle_id}'] = result_df[result_df['principle_id'] == principle_id]['prob_NA']
        else:
            result_df[f'prob_NA_p{principle_id}'] = 'NA'
    result_df = result_df.drop(columns=['prob_A', 'prob_B', 'prob_NA', 'principle_id'], errors='ignore')

    # Collapse rows across principles into a single row per prompt
    result_df = result_df.groupby(['row_id', 'order']).mean().reset_index()

    # Merge results with the rest of the columns
    if 'order' in df.columns:
        result_df = result_df.merge(df, on=['row_id', 'order'], suffixes=('', '_dupe'))
    else:
        result_df = result_df.merge(df, on=['row_id'], suffixes=('', '_dupe'))
    result_df = result_df.drop(columns=[col for col in result_df.columns if col.endswith('_dupe')])

    result_df.to_csv(config['output'].replace('.csv', '_processed.csv'), index=False)


def compose_prompts(df, principles, config):
    # If include_reversed, we will include the original comparison prompt and the version with the two options swapped
    all_orders = [0, 1] if config['include_reversed'] else [0]
    is_order_present = 1 if 'order' in df.columns else 0

    def compose_prompts_for_row(row):
        batch_prompts = []
        row_ids = []
        principle_ids = []
        orders = []

        if principles is None:
            principles = {0: "generic"}

        for principle_id, principle in principles.items():
            for order in all_orders:
                x = row['prompt']
                y_a = row['chosen'] if order == 0 else row['rejected']
                y_b = row['chosen'] if order == 1 else row['rejected']

                if principle == 'generic':
                    prompt = form_generic_chat_prompt(x, y_a, y_b, include_na=config['include_na'])
                else:
                    prompt = form_chat_prompt(x, y_a, y_b, principle, include_na=config['include_na'])
                batch_prompts.append(prompt)
                row_ids.append(row['row_id'])
                principle_ids.append(principle_id)
                orders.append(row["order"] if is_order_present else order)

        return pd.DataFrame({
            'prompt': batch_prompts,
            'row_id': row_ids,
            'principle_id': principle_ids,
            'order': orders
        })

    prompts_df = pd.concat(df.apply(compose_prompts_for_row, axis=1).tolist(), ignore_index=True)
    return prompts_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", nargs='?', default=DEFAULTS["dataset_name"])
    parser.add_argument("--principles_name", nargs='?', default=DEFAULTS["principles_name"])
    parser.add_argument("--model_name", nargs='?', default=DEFAULTS["model_name"])
    parser.add_argument("--start_row", nargs='?', type=int, default=DEFAULTS["start_row"])
    parser.add_argument("--end_row", nargs='?', type=int, default=DEFAULTS["end_row"])
    parser.add_argument("--split", nargs='?', default=DEFAULTS["split"])
    parser.add_argument("--batch_size", nargs='?', type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--torch_dtype", nargs='?', default=DEFAULTS["torch_dtype"])
    parser.add_argument("--devices", nargs='?', default=DEFAULTS["devices"])
    parser.add_argument("--include_na", action='store_true', default=DEFAULTS["include_na"])
    parser.add_argument("--include_reversed", action='store_true', default=DEFAULTS["include_reversed"])
    parser.add_argument("--continue", action='store_true', default=DEFAULTS["continue"])
    parser.add_argument("--output", nargs='?', default=DEFAULTS["output"])
    args = parser.parse_args()
    main(vars(args))
