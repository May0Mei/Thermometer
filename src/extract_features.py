import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM
)

from src.configs.options import process_args
from src.data.data_io import load_data
from src.utils import reset_seed
from src.fit_thermometer import get_logits_labels

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = process_args()

# define base LLM and tokenizer
CACHE_PATH = f"{args.root_path}.cache/huggingface/models"
if args.model_name in ['flan-t5-small', 'flan-t5-base', 'flan-t5-large','flan-t5-xl', 'flan-t5-xxl']:
    model_load_name = f'google/{args.model_name}'
    model_base = AutoModelForSeq2SeqLM.from_pretrained(model_load_name, cache_dir=CACHE_PATH)
    tokenizer = AutoTokenizer.from_pretrained(model_load_name)
elif args.model_name in ['Llama-2-7b-chat-hf']:
    print("Warning: Please provide User Access Tokens.")
    model_load_name = f'meta-llama/{args.model_name}'
    model_base = AutoModelForCausalLM.from_pretrained(model_load_name, cache_dir=CACHE_PATH, use_auth_token=args.HG_token)
    model_base.config.pad_token_id = model_base.config.eos_token_id
    tokenizer = AutoTokenizer.from_pretrained(model_load_name, padding_side='left', use_auth_token=args.HG_token)
else:
    raise NotImplementedError

reset_seed(args.training_seed) # for consistency
args.dataset = args.test_dataset
data_loader = load_data(args, tokenizer) # define the dataloader

# extract logits, labels, and final-layer features
save_path = f"./checkpoint/saved_logits/{args.dataset}/{args.model_name}"
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.isfile(os.path.join(save_path, 'features_train.pt')):
    unscaled_logits_train, true_labels_train, features_train = get_logits_labels(args.model_type, model_base, device,
                                                                tokenizer=tokenizer,
                                                                vocabulary=data_loader['target_vocabulary'],
                                                                dataloader=data_loader['train_loader'])
    unscaled_logits_val, true_labels_val, features_val = get_logits_labels(args.model_type, model_base, device,
                                                        tokenizer=tokenizer,
                                                        vocabulary=data_loader['target_vocabulary'],
                                                        dataloader=data_loader['val_loader'])
    torch.save(unscaled_logits_train.cpu().detach(), os.path.join(save_path, 'unscaled_logits_train.pt'))
    torch.save(unscaled_logits_val.cpu().detach(), os.path.join(save_path, 'unscaled_logits_val.pt'))
    torch.save(features_train.cpu().detach(), os.path.join(save_path, 'features_train.pt'))
    torch.save(true_labels_val.cpu().detach(), os.path.join(save_path, 'true_labels_val.pt'))
    torch.save(true_labels_train.cpu().detach(), os.path.join(save_path, 'true_labels_train.pt'))
    torch.save(features_val.cpu().detach(), os.path.join(save_path, 'features_val.pt'))