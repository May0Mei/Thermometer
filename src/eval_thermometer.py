import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from torch import nn
import numpy as np
import csv
import transformers
from transformers import (
    AutoTokenizer
)

from src.configs.options import process_args
from src.fit_thermometer import Thermometer, eval_thermometer
from src.data.data_io import load_thermometer_data
from src.utils import reset_seed

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = process_args()

# define the tokenizer and hidden-dim of Thermometer model
CACHE_PATH = f"{args.root_path}.cache/huggingface/models"
if args.model_name in ['flan-t5-small', 'flan-t5-base', 'flan-t5-large','flan-t5-xl', 'flan-t5-xxl']:
    model_load_name = f'google/{args.model_name}'
    tokenizer = AutoTokenizer.from_pretrained(model_load_name)
    args.thermometer_input_size = 2048
    args.thermometer_hidden_size = 256
elif args.model_name in ['Llama-2-7b-chat-hf']:
    print("Warning: Please provide User Access Tokens.")
    model_load_name = f'meta-llama/{args.model_name}'
    tokenizer = AutoTokenizer.from_pretrained(model_load_name, padding_side='left', use_auth_token=args.HG_token)
    args.thermometer_input_size = 4096
    args.thermometer_hidden_size = 512
else:
    raise NotImplementedError

# load the saved logits, labels, and features
load_path = f"{args.root_path}/checkpoint/saved_logits/{args.test_dataset}/{args.model_name}"
unscaled_logits_train = torch.load(os.path.join(load_path, 'unscaled_logits_train.pt'))
true_labels_train = torch.load(os.path.join(load_path, 'true_labels_train.pt'))
features_train = torch.load(os.path.join(load_path, 'features_train.pt'))
unscaled_logits_val = torch.load(os.path.join(load_path, 'unscaled_logits_val.pt'))
features_val = torch.load(os.path.join(load_path, 'features_val.pt'))
# define the data_loader for Thermometer
args.dataset = args.test_dataset
uq_loader = load_thermometer_data(args, tokenizer, unscaled_logits_train, unscaled_logits_val, features_train, features_val)
reset_seed(66) # To ensure consistency

# Testing
ACC_list = []
temperature_list = []
NLL_list = []
ECE_list = []
TOP_ECE_list = []
MCE_list = []
Brier_list = []
for seed in args.seed_list:
    # Define the saving path
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
    model_thermometer = Thermometer(args.thermometer_input_size, args.thermometer_hidden_size)
    model_thermometer.load_state_dict(torch.load(os.path.join(save_path, 'model_ckpt.t7')), strict=False)
    model_thermometer.to(device)
    calibration_results_thermometer = eval_thermometer(model_thermometer, device, tokenizer=tokenizer,
                                                       vocabulary=uq_loader['target_vocabulary'],
                                                       eval_dataloader=uq_loader['train_loader'])
    ACC_list.append(calibration_results_thermometer['Accuracy'])
    temperature_list.append(calibration_results_thermometer['Temperature'])
    NLL_list.append(calibration_results_thermometer['NLL'])
    ECE_list.append(calibration_results_thermometer['ECE'])
    TOP_ECE_list.append(calibration_results_thermometer['Top-ECE'])
    MCE_list.append(calibration_results_thermometer['MCE'])
    Brier_list.append(calibration_results_thermometer['Brier'])

# Save final results
csv_base_path = './results/thermometer/'
csv_path = os.path.join(
    csv_base_path,
    args.model_name,
    str(args.lambda_reg),
    str(args.test_dataset)
)
if not os.path.exists(csv_path):
    os.makedirs(csv_path)
csv_NLL = os.path.join(csv_path, 'NLL_results.csv')
csv_ECE = os.path.join(csv_path, 'ECE_results.csv')
csv_TOP_ECE = os.path.join(csv_path, 'TOP_ECE_results.csv')
csv_MCE = os.path.join(csv_path, 'MCE_results.csv')
csv_Brier = os.path.join(csv_path, 'Brier_results.csv')

columns = ["Dataset", "Test Accuracy", "Predicted Temperature", "NLL Score"]
with open(csv_NLL, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(columns)
    for i, seed in enumerate(args.seed_list):
        writer.writerow([args.test_dataset,
                        ACC_list[i],
                        temperature_list[i],
                        NLL_list[i]])
    writer.writerow([args.test_dataset,
                     f"{np.mean(ACC_list):.3f} +/- {np.std(ACC_list):.3f}",
                     f"{np.mean(temperature_list):.3f}",
                     f"{np.mean(NLL_list):.3f} +/- {np.std(NLL_list):.3f}"])

columns = ["Dataset", "Test Accuracy", "Predicted Temperature", "ECE Score"]
with open(csv_ECE, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(columns)
    for i, seed in enumerate(args.seed_list):
        writer.writerow([args.test_dataset,
                        ACC_list[i],
                        temperature_list[i],
                        ECE_list[i]])
    writer.writerow([args.test_dataset,
                     f"{np.mean(ACC_list):.3f} +/- {np.std(ACC_list):.3f}",
                     f"{np.mean(temperature_list):.3f}",
                     f"{np.mean(ECE_list):.3f} +/- {np.std(ECE_list):.3f}"])

columns = ["Dataset", "Test Accuracy", "Predicted Temperature", "TOP-ECE Score"]
with open(csv_TOP_ECE, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(columns)
    for i, seed in enumerate(args.seed_list):
        writer.writerow([args.test_dataset,
                        ACC_list[i],
                        temperature_list[i],
                        TOP_ECE_list[i]])
    writer.writerow([args.test_dataset,
                     f"{np.mean(ACC_list):.3f} +/- {np.std(ACC_list):.3f}",
                     f"{np.mean(temperature_list):.3f}",
                     f"{np.mean(TOP_ECE_list):.3f} +/- {np.std(TOP_ECE_list):.3f}"])

columns = ["Dataset", "Test Accuracy", "Predicted Temperature", "MCE Score"]
with open(csv_MCE, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(columns)
    for i, seed in enumerate(args.seed_list):
        writer.writerow([args.test_dataset,
                        ACC_list[i],
                        temperature_list[i],
                        MCE_list[i]])
    writer.writerow([args.test_dataset,
                     f"{np.mean(ACC_list):.3f} +/- {np.std(ACC_list):.3f}",
                     f"{np.mean(temperature_list):.3f}",
                     f"{np.mean(MCE_list):.3f} +/- {np.std(MCE_list):.3f}",])

columns = ["Dataset", "Test Accuracy", "Predicted Temperature", "Brier Score"]
with open(csv_Brier, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(columns)
    for i, seed in enumerate(args.seed_list):
        writer.writerow([args.test_dataset,
                        ACC_list[i],
                        temperature_list[i],
                        Brier_list[i]])
    writer.writerow([args.test_dataset,
                     f"{np.mean(ACC_list):.3f} +/- {np.std(ACC_list):.3f}",
                     f"{np.mean(temperature_list):.3f}",
                     f"{np.mean(Brier_list):.3f} +/- {np.std(Brier_list):.3f}",])



