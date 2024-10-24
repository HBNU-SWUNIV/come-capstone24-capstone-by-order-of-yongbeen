import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base_model_name', type=str, default='capstonedesignwithlyb/bllossom3.1_ga_v3')

parser.add_argument('--task', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--model_ckpt', type=str)
parser.add_argument('--token', type=str)
parser.add_argument('--device', type=str, default='0')

args = parser.parse_args()

# ag, ga, awe, mul, sentence_gec, sentence_essay_gec
task = args.task
model_id = args.base_model_name
token = args.token
output_dir = args.output_dir
device = args.device
model_ckpt = args.model_ckpt

os.environ['CUDA_VISIBLE_DEVICES'] = device

PROMPT = '''당신은 문법 오류를 수정하고 자동 글쓰기 평가를 제공하는 유용한 AI 어시스턴트입니다. 모든 답변은 정확하고 명료해야 합니다.
You are a helpful AI assistant tasked with correcting grammar errors and providing automated writing assessments. All responses must be accurate and clear.'''

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch
import pandas as pd
import random
import math
import wandb
from tqdm import tqdm

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, model_ckpt, is_trainable=False)
tokenizer = AutoTokenizer.from_pretrained(model_id)

ds = load_dataset('emotion-trash/gec_datasets', task, split='test')

result = []

model.to('cuda')

for sample in tqdm(ds, total=len(ds)):
    input_text = sample['input']
    instruction = sample['instruction']

    messages = [
        {"role": "system", "content": f"{PROMPT}"},
        {"role": "user", "content": f"{instruction}\n{input_text}"}
    ]   

    tokenized_input = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
    tokenized_input = tokenized_input.to('cuda')

    output = model.generate(tokenized_input, max_length=8192, eos_token_id=tokenizer.eos_token_id)
    output_text = tokenizer.decode(output[0][len(tokenized_input[0]):], skip_special_tokens=True)
    print(output_text)

    result.append(output_text)

import pickle


with open(f'{output_dir}.pkl', 'wb') as f:
    pickle.dump(result, f)