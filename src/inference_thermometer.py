import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM
)

from src.fit_thermometer import Thermometer

from src.configs.options import process_args
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = process_args()

# define the tokenizer and hidden-dim of Thermometer model
CACHE_PATH = f"{args.root_path}.cache/huggingface/models"
if args.model_name in ['flan-t5-small', 'flan-t5-base', 'flan-t5-large','flan-t5-xl', 'flan-t5-xxl']:
    model_load_name = f'google/{args.model_name}'
    model_base = AutoModelForSeq2SeqLM.from_pretrained(model_load_name, cache_dir=CACHE_PATH)
    tokenizer = AutoTokenizer.from_pretrained(model_load_name)
    args.thermometer_input_size = 2048
    args.thermometer_hidden_size = 256
elif args.model_name in ['Llama-2-7b-chat-hf']:
    print("Warning: Please provide User Access Tokens.")
    model_load_name = f'meta-llama/{args.model_name}'
    model_base = AutoModelForCausalLM.from_pretrained(model_load_name, cache_dir=CACHE_PATH, use_auth_token=args.HG_token)
    model_base.config.pad_token_id = model_base.config.eos_token_id
    tokenizer = AutoTokenizer.from_pretrained(model_load_name, padding_side='left', use_auth_token=args.HG_token)
    args.thermometer_input_size = 4096
    args.thermometer_hidden_size = 512
else:
    raise NotImplementedError

seed = 0

base_path = f'{args.root_path}/checkpoint/saved_thermometer/'
if args.benchmark in ['mrqa']:
    directory = args.benchmark
else:
    directory = args.test_dataset
save_path = os.path.join(
    base_path,
    directory,
    args.model_name,
    f'lambda_reg{args.lambda_reg}_seed_{seed}'
    )


# Load the model
model_thermometer = Thermometer(args.thermometer_input_size, args.thermometer_hidden_size)
model_thermometer.load_state_dict(torch.load(os.path.join(save_path, 'model_ckpt.t7')), strict=False)
model_thermometer.eval()

# Find last hidden state
model = model_base.to(device)
model.eval()
last_hidden_state = model(input_ids=source_ids,
                            attention_mask=source_mask,
                            labels=possible_labels,
                            output_hidden_states=True,
                            return_dict=True
                            ).decoder_hidden_states[-1]
temperature_inverse = nn.Softplus()(model_thermometer(last_hidden_state))