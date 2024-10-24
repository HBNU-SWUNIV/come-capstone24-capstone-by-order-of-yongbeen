import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base_model_name', type=str, default='MLP-KTLim/llama-3.1-Korean-Bllossom-8B')
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--batch', type=int, default=4)
parser.add_argument('--token', type=str)
parser.add_argument('--device', type=str, default='3')
parser.add_argument('--output_dir', type=str)
parser.add_argument('--ds_name', type=str)
parser.add_argument('--model_ckpt', type=str)
parser.add_argument('--is_bfloat16', type=bool)


args = parser.parse_args()

# ag, ga, awe, mul, sentence_gec, sentence_essay_gec
model_id = args.base_model_name
token = args.token
epoch = args.epoch
batch_size = args.batch
device = args.device
output_dir = args.output_dir
ds_name = args.ds_name
model_ckpt = args.model_ckpt
is_bfloat16 = args.is_bfloat16

ckpt_name = os.listdir(model_ckpt)[0]
model_ckpt = os.path.join(model_ckpt, ckpt_name)


os.environ['CUDA_VISIBLE_DEVICES'] = device

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch
import pandas as pd
import random
import math
import wandb


wandb.init(mode='disabled')
def fix_seed(random_seed):
    """
    fix seed to control any randomness from a code 
    (enable stability of the experiments' results.)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(random_seed)
fix_seed(42)

PROMPT = '''당신은 문법 오류를 수정하고 자동 글쓰기 평가를 제공하는 유용한 AI 어시스턴트입니다. 모든 답변은 정확하고 명료해야 합니다.
You are a helpful AI assistant tasked with correcting grammar errors and providing automated writing assessments. All responses must be accurate and clear.'''


dataset = load_dataset('emotion-trash/gec_datasets', ds_name, split='train', token=token)

model = AutoModelForCausalLM.from_pretrained('MLP-KTLim/llama-3.1-Korean-Bllossom-8B', torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, model_ckpt, is_trainable=True)

if is_bfloat16:
    for param in model.parameters():
        param.data = param.data.to(torch.bfloat16)

model.to('cuda')

tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
tokenizer.padding_side = 'right'

def formatting(batch):
    IGNORE_INDEX = -100
    max_seq_length = 30000
    train_input_ids = []
    train_label_ids = []
    
    for input_text, label, instruction in zip(batch['input'], batch['output'], batch['instruction']):
        messages = [
            {"role": "system", "content": f"{PROMPT}"},
            {"role": "user", "content": f"{instruction}\n{input_text}"}
        ]
        # print(messages)

        tokenized_input = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        tokenized_label = tokenizer(label, add_special_tokens=False, return_attention_mask=False)
        tokenized_label['input_ids'] += [tokenizer.eos_token_id]

        input_ids = (tokenized_input + tokenized_label['input_ids'])[:max_seq_length]
        label_ids = ([IGNORE_INDEX] * len(tokenized_input) + tokenized_label['input_ids'])[:max_seq_length]

        train_input_ids.append(input_ids)
        train_label_ids.append(label_ids)
    
    return {'input_ids': train_input_ids, 'labels': train_label_ids}

dataset = dataset.map(formatting, num_proc=8, batched=True, remove_columns=dataset.column_names)

class DataCollatorWithPadding(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids, labels = tuple([sample[key] for sample in batch] for key in ('input_ids', 'labels'))
        padded_input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(label) for label in labels], batch_first=True, padding_value=-100)
        attention_mask = padded_input_ids.ne(self.tokenizer.pad_token_id)

        return {'input_ids': padded_input_ids, 'labels': padded_labels, 'attention_mask': attention_mask}
    
datacollator = DataCollatorWithPadding(tokenizer)

# Training Arguments

steps_per_epoch = math.ceil(len(dataset) / batch_size)
log_steps = steps_per_epoch // 10

model_output_dir = output_dir

if not os.path.exists(model_output_dir):
    os.makedirs(model_output_dir)

training_args = TrainingArguments(
    dataloader_num_workers=8,
    output_dir = model_output_dir,
    num_train_epochs = epoch,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    save_strategy='epoch',
    save_steps=1,
    save_total_limit=1,
    optim='adamw_bnb_8bit',
    logging_strategy='epoch',
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
