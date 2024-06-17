import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, PeftModel
import torch
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset, DatasetDict
from tqdm import tqdm
import pickle
import argparse
from config import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='capstonedesignwithlyb/GEC_L1L2_30ep')
parser.add_argument('--data_path', type=str, default='capstonedesignwithlyb/sample_l2')
parser.add_argument('--save_path', type=str, default='./')

args = parser.parse_args()

model_path = args.model_path
output_path = args.save_path
data_path = args.data_path

# l1, l2, l12
dataset = load_dataset(data_path, split='train')

print(f"{model_path}model predict")

base_model_path = 'capstonedesignwithlyb/base_LLM'
model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, quantization_config=quantization_config)
model = PeftModel.from_pretrained(model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'right'

result = []
result = model_predict(model, tokenizer, dataset)

with open(f'{output_path}/output.txt', 'w') as file:
    for line in result:
        file.write(line + '\n') 

print("Data written to file.")