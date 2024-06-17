from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, PeftModel
import torch
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset, DatasetDict
from tqdm import tqdm
import pickle
import random


PROMPT_TEMPLATE = (
        "[INST] <<SYS>>\n"
        "You are a helpful assistant.\n"
        "<</SYS>>\n\n{instruction} [/INST]"
    )

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


quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
            )

peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","down_proj","up_proj"],
            r=64,
            lora_alpha=16,
            lora_dropout=0.05,
            modules_to_save=["embed_tokens","lm_head"]
            )

class DataCollatorWithPadding(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids, labels = tuple([sample[key] for sample in batch] for key in ('input_ids', 'labels'))
        padded_input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(label) for label in labels], batch_first=True, padding_value=-100)
        attention_mask = padded_input_ids.ne(self.tokenizer.pad_token_id)

        return {'input_ids': padded_input_ids, 'labels': padded_labels, 'attention_mask': attention_mask}
 

def model_predict(model, tokenizer, dataset):
    PROMPT_TEMPLATE = (
        "[INST] <<SYS>>\n"
        "You are a helpful assistant.\n"
        "<</SYS>>\n\n{instruction} [/INST]"
    )
    
    model_outputs = []
    
    for sample in tqdm(dataset, total=len(dataset)):
        if sample['input'] != '':
            messages = sample['instruction'] + '\n' + sample['input']
        else:
            messages = sample['instruction']
        
        messages = PROMPT_TEMPLATE.format(instruction=messages)
        
        tokenized_input = tokenizer(messages, return_attention_mask=False, return_tensors='pt').to(model.device)
        output = model.generate(**tokenized_input)
        
        text = tokenizer.decode(output[0])
        model_outputs.append(text)
        print(text)
    
    return model_outputs