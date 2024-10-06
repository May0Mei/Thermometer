import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
import transformers
import numpy as np
from transformers import (
    AutoTokenizer
)

from src.configs.options import process_args
from src.data.data_io import load_thermometer_data
from src.utils import reset_seed
from src.fit_thermometer import Thermometer, train_thermometer

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
model_thermometer = Thermometer(args.thermometer_input_size, args.thermometer_hidden_size)

# define the list of tasks in each benchmark
if args.benchmark == 'mmlu':
    tasks_list = ['high_school_european_history', 'business_ethics', 'clinical_knowledge',
                  'medical_genetics',
                  'high_school_us_history', 'high_school_physics', 'high_school_world_history', 'virology',
                  'high_school_microeconomics', 'econometrics', 'college_computer_science',
                  'high_school_biology',
                  'abstract_algebra', 'professional_accounting', 'philosophy', 'professional_medicine',
                  'nutrition',
                  'global_facts', 'machine_learning', 'security_studies', 'public_relations',
                  'professional_psychology',
                  'prehistory', 'anatomy', 'human_sexuality', 'college_medicine',
                  'high_school_government_and_politics',
                  'college_chemistry', 'logical_fallacies', 'high_school_geography',
                  'elementary_mathematics',
                  'human_aging', 'college_mathematics', 'high_school_computer_science', 'formal_logic',
                  'high_school_statistics', 'international_law', 'high_school_mathematics',
                  'high_school_psychology',
                  'conceptual_physics', 'miscellaneous', 'high_school_chemistry', 'marketing',
                  'professional_law',
                  'management', 'college_physics', 'jurisprudence', 'world_religions', 'sociology',
                  'us_foreign_policy',
                  'high_school_macroeconomics', 'computer_security', 'moral_disputes', 'moral_scenarios',
                  'electrical_engineering', 'astronomy', 'college_biology']
elif args.benchmark == 'bigbench':
    tasks_list = ['arithmetic', 'bbq_lite_json', 'cifar10_classification', 'color',
                  'contextual_parametric_knowledge_conflicts',
                  'elementary_math_qa', 'epistemic_reasoning', 'fact_checker', 'formal_fallacies_syllogisms_negation',
                  'goal_step_wikihow', 'hyperbaton', 'logical_fallacy_detection', 'mnist_ascii',
                  'movie_dialog_same_or_different',
                  'play_dialog_same_or_different', 'real_or_fake_text', 'social_iqa', 'strategyqa', 'timedial',
                  'tracking_shuffled_objects', 'vitaminc_fact_verification', 'unit_conversion', 'winowhy']
elif args.benchmark == 'mrqa':
    tasks_list = ['SQuAD', 'SearchQA', 'NaturalQuestionsShort', 'HotpotQA', 'NewsQA', 'TriviaQA-web']
else:
    raise NotImplementedError

# load the saved logits, labels, and features
data_loader_dict = {}
for dataset_name in tasks_list:
    print('dataset_name', dataset_name)
    args.dataset = dataset_name
    load_path = f"./checkpoint/saved_logits/{args.dataset}/{args.model_name}"
    if not os.path.exists(load_path):
        print("Warning: Please run extract_features.py first!")
    else:
        unscaled_logits_train = torch.load(os.path.join(load_path, 'unscaled_logits_train.pt'))
        features_train = torch.load(os.path.join(load_path, 'features_train.pt'))
        unscaled_logits_val = torch.load(os.path.join(load_path, 'unscaled_logits_val.pt'))
        features_val = torch.load(os.path.join(load_path, 'features_val.pt'))
    # define the data_loader for Thermometer
    data_loader = load_thermometer_data(args, tokenizer, unscaled_logits_train, unscaled_logits_val,
                                        features_train, features_val)
    # store each data_loader in a dictionary using dataset_name as the key
    data_loader_dict[dataset_name] = data_loader

# Define the saving path
base_path = './checkpoint/saved_thermometer/'
if args.test_dataset in tasks_list:
    tasks_list.remove(args.test_dataset) # leave one out
    dataset_directory = f'{args.test_dataset}/'
elif args.benchmark == 'mrqa':
    print('use all the mrqa training datasets')
    dataset_directory = f'{args.benchmark}/'
else:
    raise NotImplementedError
save_path = os.path.join(
    base_path,
    dataset_directory,
    args.model_name,
    f'lambda_reg{args.lambda_reg}_seed_{args.training_seed}'
)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# training Thermometer
torch.cuda.empty_cache() # Save GPU memory
reset_seed(args.training_seed) # For consistency
train_thermometer(model_thermometer, tasks_list, save_path, device,
            tokenizer=tokenizer,
            data_loader_dict = data_loader_dict,
            lambda_reg = args.lambda_reg,
            Gamma_k = args.Gamma_k,
            Gamma_theta = args.Gamma_theta,
            learning_rate=args.thermometer_lr,
            num_steps=args.steps)


