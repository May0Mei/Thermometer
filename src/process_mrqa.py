import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer)
import evaluate
import pandas as pd
from tqdm import tqdm
import warnings

from src.configs.options import process_args
from src.data.data_io import load_data
from src.utils import reset_seed

warnings.filterwarnings("ignore")
tqdm.disable = False
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rouge = evaluate.load('rouge')
args = process_args()
CACHE_PATH = f"{args.root_path}.cache/huggingface/models"
META_DATASET_MAX_SIZE = 50000
MAX_NEW_TOKENS = 10


def meta_prompt(generated_text):
    prompt = (f"Choose A, or B,\n {generated_text} "f"\nIs the above answer correct? \nA. Yes, \nB. No, \nAnswer: ")
    return prompt

def generate_meta_label(sentences1, sentences2):
    meta_label = "No"
    results_with_stem = rouge.compute(predictions=[sentences1], references=[sentences2], use_aggregator=False,
                                      use_stemmer=True)
    score = results_with_stem['rougeL'][0]
    if results_with_stem['rougeL'][0] > 0.:
        meta_label = "Yes"
    return meta_label, score


if args.model_name in ['Llama-2-7b-chat-hf']:
    model_load_name = f'meta-llama/{args.model_name}'
    model_base = AutoModelForCausalLM.from_pretrained(model_load_name, torch_dtype=torch.bfloat16,
                                                      cache_dir=CACHE_PATH, use_auth_token=args.HG_token).to(device)
    model_base.config.pad_token_id = model_base.config.eos_token_id
    tokenizer = AutoTokenizer.from_pretrained(model_load_name, padding_side='left', use_auth_token=args.HG_token)
    tokenizer.pad_token = tokenizer.eos_token
else:
    raise NotImplementedError

## create path for saving processed data
save_path = f"{args.root_path}.cache/huggingface/datasets"
if not os.path.exists(save_path):
    os.makedirs(save_path)
print(f"Saving processed data to {save_path}")

reset_seed(42)
args.dataset = 'mrqa'
data_loaders = load_data(args, tokenizer)
if args.dataset_split == 'train':
    dataloader = data_loaders['train_loader']
elif args.dataset_split == 'val':
    dataloader = data_loaders['val_loader']
else:
    dataloader = data_loaders['test_loader']

# create an empty meta dataset (pandas data frame)
meta_dataset = pd.DataFrame(columns=['task', 'text', 'original_label', 'generated_label', 'rougeL', 'meta_label_rouge'])
# start process the raw data
model_base.eval()
for i, batch in enumerate(tqdm(dataloader)):
    rs = model_base.generate(batch['source_ids'].to(device), attention_mask=batch['source_mask'].to(device),
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=False, temperature=1., top_p=1.)
    generated_texts = tokenizer.batch_decode(rs, skip_special_tokens=True)
    labels = batch['target_text']
    sub_tasks = batch['sub_task']
    for generated_text, label, sub_task in zip(generated_texts, labels, sub_tasks): # loop over the entire batch
        try:
            generated_label = generated_text.split("Answer:")[1]
        except:
            print("/tokenizer max length likely exceeded, skip this instance")
            break
        prompt = meta_prompt(generated_text=generated_text)
        meta_label_rouge, rougeL_score = generate_meta_label(sentences1=generated_label, sentences2=label)
        print(f"{generated_label} | {label} : {rougeL_score=}")
        dataset_entry = {
            "task": sub_task,
            "text": prompt,
            "original_label": label,
            "generated_label": generated_label,
            "rougeL": rougeL_score,
            "meta_label_rouge": meta_label_rouge
        }
        meta_dataset = meta_dataset.append(dataset_entry, ignore_index=True)
    if i > META_DATASET_MAX_SIZE:
        break
    if i % 20 == 0:
        meta_dataset.to_csv(f'{save_path}/{args.dataset}_meta_data_{args.dataset_split}.csv', index=False)

meta_dataset.to_csv(f'{save_path}/{args.dataset}_meta_data_{args.dataset_split}.csv', index=False)