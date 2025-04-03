from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd


def tokenize_prompts(prompts_df, tokenizer, config):
    tokenized_data = {"input_ids": [], "attention_mask": []}

    max_prompt_length = config['max_length']
    prompts = []
    for prompt in prompts_df['prompt']:
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    prompt_inputs = tokenizer(prompts, return_tensors='pt', truncation=True)
    tokenized_data['input_ids'].extend(prompt_inputs['input_ids'])
    tokenized_data['attention_mask'].extend(prompt_inputs['attention_mask'])

    tokenized_data = {
        'input_ids': torch.stack(tokenized_data["input_ids"]),
        'attention_mask': torch.stack(tokenized_data["attention_mask"]),
        'row_ids': torch.tensor(prompts_df['row_id'].values),
        'principle_ids': torch.tensor(prompts_df['principle_id'].values),
        'order': torch.tensor(prompts_df['order'].values)
    }
    return tokenized_data


class TokenizedPromptDataset(Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data["input_ids"]
        self.attention_mask = tokenized_data["attention_mask"]
        self.row_ids = tokenized_data["row_ids"]
        self.principle_ids = tokenized_data["principle_ids"]
        self.order = tokenized_data["order"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "row_id": self.row_ids[idx],
            "principle_id": self.principle_ids[idx],
            "order": self.order[idx]
        }


def annotate_prompts(prompts_df, config, existing_result_df=None):
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'])
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenized_prompts = tokenize_prompts(prompts_df, tokenizer, config)
    dataset = TokenizedPromptDataset(tokenized_prompts)

    devices = config['devices']

    torch_dtype = config['torch_dtype']
    if (len(devices) == 1) and devices[0] == 'cpu':
        torch_dtype = torch.float32

    target_token_A = "A"
    target_token_B = "B"
    target_token_id_A = tokenizer.encode(target_token_A, add_special_tokens=False)[0]
    target_token_id_B = tokenizer.encode(target_token_B, add_special_tokens=False)[0]
    if config['include_na']:
        target_token_id_NA = tokenizer.encode("NA", add_special_tokens=False)[0]

    model = AutoModelForCausalLM.from_pretrained(config['model_id'], torch_dtype=torch_dtype, device_map=devices)
    model.eval()
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    full_score_df = existing_result_df
    for batch in tqdm(loader, desc=f"Processing Batches on {devices}"):
        row_ids = batch["row_id"]
        principle_ids = batch["principle_id"]
        orders = batch["order"]
        first_device = next(model.parameters()).device
        inputs = {key: val.to(first_device) for key, val in batch.items() if key in ["input_ids", "attention_mask"]}

        with torch.no_grad():
            outputs = model(**inputs, sample=False)  # should we have hyperparameters?
            logits = outputs.logits

        last_non_padding_idx = inputs["attention_mask"].sum(dim=1).sub(1)
        next_token_logits = logits[torch.arange(logits.size(0)), last_non_padding_idx, :]
        probs = torch.softmax(next_token_logits, dim=-1)

        scores_batch_A = probs[:, target_token_id_A].cpu()
        scores_batch_B = probs[:, target_token_id_B].cpu()
        scores_df = pd.DataFrame({
            'row_id': row_ids.numpy(),
            'principle_id': principle_ids.numpy(),
            'order': orders.numpy(),
            'prob_A': scores_batch_A.numpy(),
            'prob_B': scores_batch_B.numpy()
        })
        if config['include_na']:
            scores_batch_NA = probs[:, target_token_id_NA].cpu()
            scores_df['prob_NA'] = scores_batch_NA.numpy()

        if full_score_df is None:
            full_score_df = scores_df
            scores_df.to_csv(config['output'], index=False)
        else:
            full_score_df = pd.concat([full_score_df, scores_df], ignore_index=True)
            scores_df.to_csv(config['output'], mode='a', index=False, header=False)

    return full_score_df
