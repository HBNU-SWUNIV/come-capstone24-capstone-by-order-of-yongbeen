import os

os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, get_peft_model
import torch
import pandas as pd
import random
import math
import wandb
import argparse
from config import *

fix_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path', type=str, default='capstonedesignwithlyb/sample_l2')
parser.add_argument('--eval_data_path', type=str, default='capstonedesignwithlyb/sample_l2')
parser.add_argument('--n_epoch', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--output_dir', type=str, default='./')
parser.add_argument('--do_eval', type=bool, default=False)

args = parser.parse_args()
wandb.init(mode='disabled')

dataset = load_dataset(args.train_data_path, split='train')
eval_dataset = load_dataset(args.eval_data_path, split='train')

model_id = 'capstonedesignwithlyb/base_LLM'
model = AutoModelForCausalLM.from_pretrained(model_id)
model = get_peft_model(model_id, peft_config)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'right'

def formatting(batch):
    IGNORE_INDEX = -100
    max_seq_length = 1024
    train_input_ids = []
    train_label_ids = []
    for input_text, label, instruction in zip(batch['input'], batch['output'], batch['instruction']):
        if input_text != '':
            messages = instruction + '\n' + input_text
        else:
            messages = instruction
        
        messages = PROMPT_TEMPLATE.format(instruction=messages)
        
        tokenized_input = tokenizer(messages, return_attention_mask=False)
        tokenized_label = tokenizer(label, return_attention_mask=False, add_special_tokens=False)
        
        tokenized_label['input_ids'] += [tokenizer.eos_token_id]

        input_ids = (tokenized_input['input_ids'] + tokenized_label['input_ids'])[:max_seq_length]
        label_ids = ([IGNORE_INDEX] * len(tokenized_input['input_ids']) + tokenized_label['input_ids'])[:max_seq_length]

        train_input_ids.append(input_ids)
        train_label_ids.append(label_ids)
    
    return {'input_ids': train_input_ids, 'labels': train_label_ids}

dataset = dataset.map(formatting, num_proc=8, batched=True, remove_columns=dataset.column_names)
    
datacollator = DataCollatorWithPadding(tokenizer)

# Training Arguments
epoch = args.n_epoch
batch_size = args.batch_size
output_dir=args.output_dir

steps_per_epoch = math.ceil(len(dataset)/batch_size)
log_steps = steps_per_epoch // 10

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

training_args = TrainingArguments(
    dataloader_num_workers=8,
    output_dir = output_dir,
    num_train_epochs = epoch,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    save_strategy='epoch',
    save_steps=1,
    save_total_limit=epoch,
    optim='adamw_bnb_8bit',
    logging_strategy='steps',
    logging_steps=log_steps,
    report_to=None
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=datacollator,
)

trainer.train()

if args.do_eval:
    model_result = model_predict(model, tokenizer, eval_dataset)
    with open(f'{args.output_dir}/eval.txt', 'wb') as f:
        pickle.dump(model_result, f)