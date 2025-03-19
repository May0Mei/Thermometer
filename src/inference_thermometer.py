#%% Imports
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from torch import nn
import numpy as np
import plotly.express as px

torch.cuda.empty_cache()
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM
)

from src.fit_thermometer import Thermometer
from src.configs.config import cfg as args

#%% CUDA memory, parse args
def print_cuda_memory():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%% define the tokenizer and hidden-dim of Thermometer model
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

# model checkpoint path
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


if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

#%% Load models

model = model_base
model.eval()
for param in model.parameters():
    param.requires_grad = False
model.to(device)

# Load the model
model_thermometer = Thermometer(args.thermometer_input_size, args.thermometer_hidden_size)
model_thermometer.load_state_dict(torch.load(os.path.join(save_path, 'model_ckpt.t7')), strict=False)
model_thermometer.eval()

def get_features(prompt, vocab=['Yes', 'No']):
    # Prepare input
    source = tokenizer(prompt, 
                        max_length=args.max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors='pt',
                        return_attention_mask=True)
    source_ids = source['input_ids'].squeeze().to(dtype=torch.long)
    source_mask = source['attention_mask'].squeeze().to(dtype=torch.long)
    encoded_vocab = [torch.LongTensor(tokenizer.encode(v, add_special_tokens=False)).to(device) for v in vocab]

    # Find last hidden state
    logits_list = [[] for v in encoded_vocab]
    feature_list = []

    if args.model_type == 'decoder_only':
        
        outputs = model(input_ids=source_ids.unsqueeze(axis=0).to(device),
                        attention_mask=source_mask.unsqueeze(axis=0).to(device),
                        output_hidden_states=True,
                        return_dict=True)
        logits = outputs.logits[:, -1, :]
        for i, ev in enumerate(encoded_vocab):
            logits_list[i].append(logits[:, ev].cpu().detach().numpy())
        feature_list.append(outputs.hidden_states[-1][:, -1, :].cpu().detach().numpy())
    features = torch.tensor(np.concatenate(feature_list, axis=0)).squeeze()
    return features, logits_list

def get_scaled_logits(features, logits_list):
    temperature_inverse = nn.Softplus()(model_thermometer(features.unsqueeze(dim=0)))
    temperature_inverse = temperature_inverse.cpu().detach().numpy()
    print(f"Predicted temperature: {temperature_inverse}")
    # scale the logits by predicted temperature
    logits_all = np.concatenate(logits_list, axis=0)
    logits_scaled = torch.tensor(logits_all*temperature_inverse).to(device)
    return logits_scaled

def generate_prompt(prob_yes):
    prompt = f"""The probability of observing B given A is {prob_yes},\
                 the probability of observing C given A is {1-prob_yes}. \
                 Now, we are given A. The next observation is:  """
    return prompt

def main(n):
    true_probs = np.random.rand(n)
    predicted_probs = np.zeros_like(true_probs)
    for i, prob_yes in enumerate(true_probs):
        prompt = generate_prompt(prob_yes)
        features, logits_list = get_features(prompt)
        logits_scaled = get_scaled_logits(features, logits_list)
        prob_yes_pred = torch.nn.functional.softmax(logits_scaled, dim=0)[0, 0].item()
        predicted_probs[i] = prob_yes_pred
        
        print(f"True probability: {prob_yes:.3f}, Scaled logits: {logits_scaled}")
    
    # make a calibration plot with plotly
    fig = px.scatter(x=true_probs, y=predicted_probs, labels={'x':'True probability', 'y':'Predicted probability'})
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_layout(title='Calibration plot using Thermometer model')
    # save figure
    fig.write_html(f"calibration_plot.html")
    




if __name__ == '__main__':
    main(10)