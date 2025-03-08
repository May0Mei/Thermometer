import torch
from torch.utils.data import DataLoader, Dataset
from src.data.data_utils import TASK_MAPPING
from rich import print

mrqa_vocab = {"Yes": "A", "No": "B"}
mmlu_vocab = {'A' : 'A', 'B': 'B', 'C': 'C', 'D': 'D'}
# set target to be multiple choices letters like "A","B","C" to ensure consistency
VOCAB_MAPPING = {
    ############################ bigbench dataset ############################
    "arithmetic": {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G'},
    "bbq_lite_json": {'0' : 'A', '1': 'B', '2': 'C' },
    "cifar10_classification": {'0' : 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I', '9': 'J'},
    "contextual_parametric_knowledge_conflicts": {'0' : 'A', '1': 'B'},
    "color": {'0' : 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I', '9': 'J'},
    "elementary_math_qa": {'0' : 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E'},
    "epistemic_reasoning": {"0": "A", "1": "B"},
    "fact_checker": {"0": "A", "1": "B"},
    "formal_fallacies_syllogisms_negation": {"0": "A", "1": "B"},
    "goal_step_wikihow": {'0' : 'A', '1': 'B', '2': 'C', '3': 'D'},
    "hyperbaton": {"0": "A", "1": "B"},
    "logical_fallacy_detection": {"0": "A", "1": "B"},
    "mnist_ascii": {'0' : 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I', '9': 'J'},
    "movie_dialog_same_or_different": {"0": "A", "1": "B"},
    "play_dialog_same_or_different": {"0": "A", "1": "B"},
    "real_or_fake_text": {'0' : 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I', '9': 'J'},
    "social_iqa": {'0' : 'A', '1': 'B', '2': 'C' },
    "strategyqa": {"0": "A", "1": "B"},
    "timedial": {'0': 'A', '1': 'B', '2': 'C'},
    "tracking_shuffled_objects": {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G'},
    "vitaminc_fact_verification": {'0': 'A', '1': 'B', '2': 'C'},
    "unit_conversion": {'0' : 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E'},
    "winowhy": {"0": "A", "1": "B"},
    ############################ mmlu datasets ############################
    "abstract_algebra": mmlu_vocab,
    "anatomy":  mmlu_vocab,
    "astronomy":  mmlu_vocab,
    "college_biology":  mmlu_vocab,
    "electrical_engineering":  mmlu_vocab,
    "moral_disputes":  mmlu_vocab,
    "moral_scenarios":  mmlu_vocab,
    "computer_security":  mmlu_vocab,
    "high_school_macroeconomics":  mmlu_vocab,
    "us_foreign_policy": mmlu_vocab,
    "sociology":  mmlu_vocab,
    "world_religions":  mmlu_vocab,
    "jurisprudence":  mmlu_vocab,
    "college_physics":  mmlu_vocab,
    "management":  mmlu_vocab,
    "professional_law":  mmlu_vocab,
    "marketing":  mmlu_vocab,
    "high_school_chemistry":  mmlu_vocab,
    "miscellaneous":  mmlu_vocab,
    "conceptual_physics":  mmlu_vocab,
    "high_school_computer_science":  mmlu_vocab,
    "high_school_mathematics":  mmlu_vocab,
    "international_law":  mmlu_vocab,
    "high_school_statistics":  mmlu_vocab,
    "formal_logic":  mmlu_vocab,
    "high_school_psychology":  mmlu_vocab,
    "college_mathematics": mmlu_vocab,
    "human_aging":  mmlu_vocab,
    "elementary_mathematics":  mmlu_vocab,
    "high_school_geography":  mmlu_vocab,
    "logical_fallacies":  mmlu_vocab,
    "college_chemistry":  mmlu_vocab,
    "high_school_government_and_politics":  mmlu_vocab,
    "college_medicine":  mmlu_vocab,
    "human_sexuality":  mmlu_vocab,
    "prehistory": mmlu_vocab,
    "professional_psychology": mmlu_vocab,
    "public_relations":  mmlu_vocab,
    "security_studies":  mmlu_vocab,
    "machine_learning":  mmlu_vocab,
    "global_facts":  mmlu_vocab,
    "professional_medicine":  mmlu_vocab,
    "nutrition":  mmlu_vocab,
    "philosophy": mmlu_vocab,
    "professional_accounting":  mmlu_vocab,
    "high_school_biology":  mmlu_vocab,
    "philosophy":  mmlu_vocab,
    "college_computer_science":  mmlu_vocab,
    "econometrics":  mmlu_vocab,
    "high_school_microeconomics": mmlu_vocab,
    "virology":  mmlu_vocab,
    "high_school_world_history":  mmlu_vocab,
    "high_school_physics":  mmlu_vocab,
    "high_school_us_history":  mmlu_vocab,
    "medical_genetics":  mmlu_vocab,
    "clinical_knowledge":  mmlu_vocab,
    "business_ethics":  mmlu_vocab,
    "high_school_european_history":  mmlu_vocab,
    ############################ mrqa datasets ############################
    # Train and Val
    "SQuAD": mrqa_vocab,
    "SearchQA": mrqa_vocab,
    "NaturalQuestionsShort": mrqa_vocab,
    "HotpotQA": mrqa_vocab,
    "NewsQA": mrqa_vocab,
    "TriviaQA-web": mrqa_vocab,
    # Test
    "BioASQ":mrqa_vocab,
    "DROP": mrqa_vocab,
    "DuoRC.ParaphraseRC": mrqa_vocab,
    "RACE": mrqa_vocab,
    "RelationExtraction": mrqa_vocab,
    "TextbookQA": mrqa_vocab,
    # raw data
    "mrqa": None,
}

class UqAutoTask:
    @classmethod
    def get(self, task, config='en', seed=42):
        if task in TASK_MAPPING:
            return TASK_MAPPING[task](config, task, seed, VOCAB_MAPPING[task])
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )

class SimpleT5Dataset(Dataset):
    """
    A simple dataset for T5 or other encoder-decoder model
    """
    def __init__(self, dataframe, tokenizer, source_len, target_len, unscaled_logits=[], features = []):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.target_len = target_len
        self.unscaled_logits = unscaled_logits
        self.features = features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        source_text = self.data[index]['source']
        target_text = self.data[index]['target']

        source = self.tokenizer(source_text,
                                max_length=self.source_len,
                                padding='max_length',
                                truncation=True,
                                return_tensors='pt'
                                )

        target = self.tokenizer(target_text,
                                max_length=self.target_len,
                                padding='max_length',
                                truncation=True,
                                return_tensors='pt'
                                )

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids']
        target_ids = torch.tensor([
            [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in target_ids
        ]).squeeze()

        if len(self.unscaled_logits)!=0:
            unscaled_logits = self.unscaled_logits[index]
        else:
            unscaled_logits = self.unscaled_logits
        if len(self.features)!=0:
            features = self.features[index]
        else:
            features = self.features

        return {
                'source_ids': source_ids.to(dtype=torch.long),
                'source_mask': source_mask.to(dtype=torch.long),
                'target_ids': target_ids.to(dtype=torch.long),
                'extra_fields': str(self.data[index]['extra_fields']) if 'extra_fields' in self.data[index] else "{}",
                'unscaled_logits': unscaled_logits,
                'features':features
        }

class SimpleLLaMADataset(Dataset):
    """
    A simple dataset for LLaMA or other decoder only model
    """
    def __init__(self, dataframe, tokenizer, max_length, unscaled_logits=[], features = []):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_length = max_length
        self.unscaled_logits = unscaled_logits
        self.features = features
        if getattr(self.tokenizer, "pad_token_id") is None:
            self.tokenizer.pad_token_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        source_text = self.data[index]['source']
        target_text = self.data[index]['target']
        sub_task = []
        if 'extra_fields' in self.data[index]:
            sub_task = self.data[index]['extra_fields']["sub_task"]
        source = self.tokenizer(source_text, max_length=self.max_length, padding='max_length', truncation=True,
                                       return_tensors="pt", return_attention_mask=True)
        target = self.tokenizer(target_text, max_length=self.max_length, padding='max_length',  truncation=True, return_tensors="pt")

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids']
        target_ids = torch.tensor([[(l if l != self.tokenizer.pad_token_id else -100) for l in label]
                                   for label in target_ids]).squeeze()

        if len(self.unscaled_logits)!=0:
            unscaled_logits = self.unscaled_logits[index]
        else:
            unscaled_logits = self.unscaled_logits
        if len(self.features)!=0:
            features = self.features[index]
        else:
            features = self.features

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_text': target_text,
            'target_ids': target_ids.to(dtype=torch.long),
            'unscaled_logits': unscaled_logits,
            'features': features,
            'sub_task': sub_task
        }


def load_data(args, tokenizer):
    """
    :param args:
    :param tokenizer:
    :return: A dictionary of dataloaders and target_vocabulary:
    {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader
        'target_vocabulary': target_vocab
    }
    """
    dataset_name = args.dataset

    train_dataset = UqAutoTask.get(dataset_name).get(split="train",
                                                   split_validation_test=True,
                                                   file_prefix=args.root_path,
                                                   n_obs=4000)

    val_dataset = UqAutoTask.get(dataset_name).get(split="validation",
                                                 split_validation_test=True,
                                                 file_prefix=args.root_path,
                                                 n_obs=1000) 

    test_dataset = UqAutoTask.get(dataset_name).get(split="test",
                                                  split_validation_test=True,
                                                  file_prefix=args.root_path,
                                                  n_obs=None)
    # store target vocabulary
    target_vocab = [tgt for tgt in set(train_dataset['target'])]
    if VOCAB_MAPPING[dataset_name] is not None:
        target_vocab = VOCAB_MAPPING[dataset_name]

    # Data loader
    if args.model_type == 'encoder_decoder':
        training_set = SimpleT5Dataset(train_dataset, tokenizer,
                                       args.max_source_length,
                                       args.max_target_length,
                                       )
        val_set = SimpleT5Dataset(val_dataset, tokenizer,
                                  args.max_source_length,
                                  args.max_target_length,
                                  )
        test_set = SimpleT5Dataset(test_dataset, tokenizer,
                                   args.max_source_length,
                                   args.max_target_length,
                                   )
    elif args.model_type == 'decoder_only':
        training_set = SimpleLLaMADataset(train_dataset, tokenizer,
                                       args.max_length
                                       )
        val_set = SimpleLLaMADataset(val_dataset, tokenizer,
                                  args.max_length
                                  )
        test_set = SimpleLLaMADataset(test_dataset, tokenizer,
                                   args.max_length
                                   )
    else:
        raise NotImplementedError
    print(dataset_name, 'training data example\t', train_dataset[0])

    # Defining the parameters for creation of dataloaders
    train_params = {
                    'batch_size': args.inference_batch_size,
                    'shuffle': False, # only for inference the base model, no shuffle
                    'num_workers': 0,
    }
    eval_params = {
                    'batch_size': args.inference_batch_size,
                    'shuffle': False,
                    'num_workers': 0,
    }
    train_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **eval_params)
    test_loader = DataLoader(test_set, **eval_params)

    return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'target_vocabulary': target_vocab
    }

def load_thermometer_data(args, tokenizer, scaled_prob_train, scaled_prob_val, features_train, features_val):
    """
    :param args:
    :param tokenizer:
    :return: the dataloader for thermometer
    """
    dataset_name = args.dataset
    
    train_dataset = UqAutoTask.get(dataset_name).get(split="train",
                                                     split_validation_test=True,
                                                     file_prefix=args.root_path,
                                                     n_obs=4000)

    val_dataset = UqAutoTask.get(dataset_name).get(split="validation",
                                                 split_validation_test=True,
                                                 file_prefix=args.root_path,
                                                 n_obs=1000)

    test_dataset = UqAutoTask.get(dataset_name).get(split="test",
                                                  split_validation_test=True,
                                                  file_prefix=args.root_path,
                                                  n_obs=None)

    # store target vocabulary
    target_vocab = [tgt for tgt in set(val_dataset['target'])]
    if VOCAB_MAPPING[dataset_name] is not None:
        target_vocab = VOCAB_MAPPING[dataset_name]

        # Data loader
        if args.model_type == 'encoder_decoder':
            training_set = SimpleT5Dataset(train_dataset, tokenizer,
                                           args.max_source_length,
                                           args.max_target_length,
                                           scaled_prob_train,
                                           features_train
                                           )
            val_set = SimpleT5Dataset(val_dataset, tokenizer,
                                      args.max_source_length,
                                      args.max_target_length,
                                      scaled_prob_val,
                                      features_val
                                      )
            test_set = SimpleT5Dataset(test_dataset, tokenizer,
                                       args.max_source_length,
                                       args.max_target_length,
                                       )
        elif args.model_type == 'decoder_only':
            training_set = SimpleLLaMADataset(train_dataset, tokenizer,
                                              args.max_length,
                                              scaled_prob_train,
                                              features_train
                                              )
            val_set = SimpleLLaMADataset(val_dataset, tokenizer,
                                         args.max_length,
                                         scaled_prob_val,
                                         features_val
                                         )
            test_set = SimpleLLaMADataset(test_dataset, tokenizer,
                                          args.max_length
                                          )
        else:
            raise NotImplementedError
    # Defining the parameters for creation of dataloaders
    train_params = {
                    'batch_size': args.train_batch_size,
                    'shuffle': True, # used to train the Thermometer model, set True
                    'num_workers': 0,
    }
    eval_params = {
                    'batch_size': args.eval_batch_size,
                    'shuffle': False,
                    'num_workers': 0,
    }
    train_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **eval_params)
    test_loader = DataLoader(test_set, **eval_params)

    return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'target_vocabulary': target_vocab
    }

