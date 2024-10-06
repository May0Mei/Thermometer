import os
import re
import abc
import logging
import functools
import numpy as np
import collections
from collections import OrderedDict
from typing import Callable, List, Mapping, Dict
import torch
import datasets
from datasets import Dataset
import pandas as pd

from src.configs.options import process_args
from src.utils import clean_meta_data
from datasets.utils import disable_progress_bar

disable_progress_bar()
logger = logging.getLogger(__name__)
args = process_args()
CACHE_PATH = f"{args.root_path}.cache/huggingface/datasets"

class BaseAbstractTask(abc.ABC):
    name = NotImplemented
    config = NotImplemented
    prefix = NotImplemented
    preprocessor: Callable = NotImplemented
    split_map = None
    labels_list = None
    split_to_data_split: Mapping[str, str] = {"train": "train", "validation": "validation", "test": "test"}

    def __init__(self, config, name, seed=42):
        self.config = config
        self.name = name
        self.seed = seed

    def seq2seq_format(self, sources: List[str],
                       targets: List[str],
                       add_prefix: bool = False,
                       prefix: str = None,
                       extra_fields={},
                       verbalizer=""):
        src_prefix = self.name if prefix is None else prefix
        if verbalizer:
            sources = [verbalizer] + sources
        sources = [src_prefix] + sources if add_prefix else sources
        if len(extra_fields) == 0:
            return {'source': ' '.join(sources),
                    'target': ' '.join(targets),
                    'task': self.name}
        else:
            return {'source': ' '.join(sources),
                    'target': ' '.join(targets),
                    'task': self.name,
                    'extra_fields': extra_fields}

    def check_n_obs(self, n_obs, total_size):
        if n_obs is not None and n_obs > total_size:
            n_obs = total_size
            logger.warning("n_obs is set to %s", n_obs)
        return n_obs

    def shuffled_indices(self, dataset):
        num_samples = len(dataset)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return torch.randperm(num_samples, generator=generator).tolist()

    def subsample(self, dataset, n_obs=None, indices=None):
        """
        Given a dataset returns the subsampled dataset.
        :param n_obs: the number of samples of the subsampled dataset.
        :param indices: indices to select the samples from, if not given, indices are computed
        from by shuffling the given dataset.
        :return: subsampled dataset.
        """
        num_samples = len(dataset)
        n_obs = self.check_n_obs(n_obs, num_samples)
        if indices is None:
            indices = self.shuffled_indices(dataset)
        indices = indices[:n_obs]
        return dataset.select(indices)

    def load_dataset(self, split: int):
        return datasets.load_dataset(self.name, self.config, split=split, cache_dir =CACHE_PATH)

    def get_split_indices(self, split, dataset, train_size, max_size):
        indices = self.shuffled_indices(dataset)
        if split == "train":
            return indices[:train_size]
        elif split == "validation":
            return indices[train_size:max_size]

    def map_dataset(self, dataset, add_prefix, add_vb):
        return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix, add_vb=add_vb),
                           remove_columns=dataset.column_names, load_from_cache_file=False)

    def get(self, split, add_prefix=True,
            n_obs=None, split_validation_test=False, lang=None, file_name=None, add_vb=False, file_prefix=None):
        self.file_prefix = file_prefix  # path prefix for external dataset loading
        mapped_split = self.split_to_data_split[split]
        if lang is not None:
            dataset = self.load_dataset(split=mapped_split, lang_code=lang)

        if file_name is not None:
            dataset = datasets.load_dataset(
                'csv', data_files=file_name, split="train")
        else:
            dataset = self.load_dataset(split=mapped_split)
        # shuffles the data and samples it.
        if n_obs is not None:
            dataset = self.subsample(dataset, n_obs)

        return self.map_dataset(dataset, add_prefix, add_vb)

class AbstractTask(BaseAbstractTask):
    def __init__(self,
                 config,
                 name,
                 seed=42,
                 vocab_mapping: Dict = None):
        super().__init__(config=config, name = name, seed=seed)
        self.vocab_mapping = vocab_mapping

    def seq2seq_format(self, sources: List[str],
                       targets: List[str],
                       add_prefix: bool = False,
                       prefix: str = None,
                       extra_fields={},
                       verbalizer=""):
        src_prefix = self.name if prefix is None else prefix
        if verbalizer:
            sources = [verbalizer] + sources
        sources = [src_prefix] + sources if add_prefix else sources
        tgt = ' '.join(targets)
        if self.vocab_mapping is not None:
            if tgt=='-1':
                tgt='0' #edge case for some missing label
            tgt = self.vocab_mapping[tgt]
        if len(extra_fields) == 0:
            return {'source': ' '.join(sources),
                    'target': tgt,
                    'task': self.name}
        else:
            return {'source': ' '.join(sources),
                    'target': tgt,
                    'task': self.name,
                    'extra_fields': extra_fields}


####################################################################################
'''
mmlu datasets
append multiple-choices answers to the questions
'''
####################################################################################
class mmlu(AbstractTask):
    labels_list = ['A', 'B', 'C', 'D']
    split_to_data_split = {"train": "test",
                           "validation": "validation",
                           "test": "train"}
    def load_dataset(self, split):
        datasets_all =  datasets.load_dataset('lukaemon/mmlu', self.name, split=split, cache_dir =CACHE_PATH)
        return datasets_all

    def preprocessor(self, example, add_prefix=True, add_vb=False):
        src_texts = ["\nChoose A, B, C, or D.",
                     "\nQuestion:{}.".format(example["input"]),
                     "\nA. {}.".format(example["A"]) ,
                     "\nB. {}.".format(example["B"]) ,
                     "\nC. {}.".format(example["C"]) ,
                     "\nD. {}.".format(example["D"]) ,
                     "\nAnswer:"]
        tgt_texts = [example["target"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)

####################################################################################
'''
bigbench datasets
append multiple-choices answers to the questions
'''
####################################################################################

class bigbench(AbstractTask):
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "train"}

    def load_dataset(self, split):
        if self.name in ['winowhy', 'strategyqa', 'play_dialog_same_or_different', 'movie_dialog_same_or_different',
                         'logical_fallacy_detection', 'contextual_parametric_knowledge_conflicts', 'epistemic_reasoning',
                         'fact_checker', 'formal_fallacies_syllogisms_negation', 'hyperbaton']:
            self.labels_list = ['A', 'B']
        elif self.name in ['bbq_lite_json', 'social_iqa', 'timedial', 'vitaminc_fact_verification']:
            self.labels_list = ['A', 'B', 'C']
        elif self.name in ['goal_step_wikihow']:
            self.labels_list = ['A', 'B', 'C', 'D']
        elif self.name in ['elementary_math_qa', 'unit_conversion']:
            self.labels_list = ['A', 'B', 'C', 'D', 'E']
        elif self.name in ['arithmetic', 'tracking_shuffled_objects']:
            self.labels_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        elif self.name in ['cifar10_classification', 'color', 'mnist_ascii', 'real_or_fake_text']:
            self.labels_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        else:
            print('Invalid Argument!')

        datasets_all =  datasets.load_dataset('tasksource/bigbench', self.name, split=split, cache_dir =CACHE_PATH)
        filtered_dataset = datasets_all.filter(lambda example: len(example["multiple_choice_scores"])==len(self.labels_list))
        return filtered_dataset

    def preprocessor(self, example, add_prefix=True, add_vb=False):
        if self.name in ['winowhy', 'strategyqa', 'play_dialog_same_or_different', 'movie_dialog_same_or_different',
                         'logical_fallacy_detection', 'contextual_parametric_knowledge_conflicts', 'epistemic_reasoning',
                         'fact_checker', 'formal_fallacies_syllogisms_negation',  'hyperbaton']:
            src_texts = ["\nChoose A, or B."]
        elif self.name in ['bbq_lite_json', 'social_iqa', 'timedial', 'vitaminc_fact_verification']:
            src_texts = ["\nChoose A, B or C."]
        elif self.name in ['goal_step_wikihow']:
            src_texts = ["\nChoose A, B, C, or D."]
        elif self.name in ['elementary_math_qa', 'unit_conversion']:
            src_texts = ["\nChoose A, B, C, D, or E."]
        elif self.name in ['arithmetic', 'tracking_shuffled_objects']:
            src_texts = ["\nChoose A, B, C, D, E, F, or G."]
        elif self.name in ['cifar10_classification', 'color', 'mnist_ascii', 'real_or_fake_text']:
            src_texts = ["\nChoose A, B, C, D, E, F, G, H, I, or J."]
        else:
            raise NotImplementedError

        if 'Answer:' in example["inputs"]:
            question = example["inputs"].split('Answer:')[0]
        elif 'choice:' in example["inputs"]:
            question = example["inputs"].split('choice:')[0]
        else:
            question = example["inputs"]
        src_texts += ["\nQuestion:{}.".format(question)]
        for i in range(len(self.labels_list)):
            src_texts += ["\n{}. {}.".format(self.labels_list[i], example["multiple_choice_targets"][i])]
        src_texts += ["\nAnswer:"]

        tgt_texts = [str(example["multiple_choice_scores"].index(1))]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


####################################################################################
'''
MRQA datasets
The model is asked to answer if its response is correct or wrong.
'''
####################################################################################
class mrqa(AbstractTask):
    labels_list = ['A', 'B']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    def load_dataset(self, split):
        file_path = os.path.join(CACHE_PATH, f'mrqa/mrqa_meta_data_{split}.csv')
        df = pd.read_csv(file_path)
        filtered_df = df[df['task'] == self.name]
        datasets_all = Dataset.from_pandas(filtered_df)
        return datasets_all

    def preprocessor(self, example, add_prefix=True, add_vb=False):
        src_texts = [example["text"]]
        tgt_texts = [example["meta_label_rouge"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)

class mrqa_test(AbstractTask):
    labels_list = ['A', 'B']
    split_to_data_split = {"train": "test",
                           "validation": "test",
                           "test": "test"}
    def load_dataset(self, split):
        file_path = os.path.join(CACHE_PATH, f'mrqa/mrqa_meta_data_{split}.csv')
        df = pd.read_csv(file_path)
        filtered_df = df[df['task'] == self.name]
        datasets_all = Dataset.from_pandas(filtered_df)
        return datasets_all

    def preprocessor(self, example, add_prefix=True, add_vb=False):
        src_texts = [example["text"]]
        tgt_texts = [example["meta_label_rouge"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)

class mrqa_raw(AbstractTask):
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}
    def load_dataset(self, split):
        datasets_all = datasets.load_dataset(self.name,  split=split, cache_dir=CACHE_PATH)
        return datasets_all

    def preprocessor(self, example, add_prefix=False, add_vb=False):
        src_texts = ["Answer in as few words as possible."
                     "\nContext:{}.".format(example["context"]),
                     "\nQuestion:{}.".format(example["question"]),
                     "\nAnswer:"]
        tgt_texts = clean_meta_data(example["answers"])
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix=False, extra_fields={"sub_task": example["subset"]})


TASK_MAPPING = OrderedDict(
    [
        ############################ mmlu datasets ############################
        ('abstract_algebra', mmlu),
        ('anatomy', mmlu),
        ('astronomy', mmlu),
        ('college_biology', mmlu),
        ('electrical_engineering', mmlu),
        ('moral_disputes', mmlu),
        ('moral_scenarios', mmlu),
        ('computer_security', mmlu),
        ('high_school_macroeconomics', mmlu),
        ('us_foreign_policy', mmlu),
        ('sociology', mmlu),
        ('world_religions', mmlu),
        ('jurisprudence', mmlu),
        ('college_physics', mmlu),
        ('management', mmlu),
        ('professional_law', mmlu),
        ('marketing', mmlu),
        ('high_school_chemistry', mmlu),
        ('miscellaneous', mmlu),
        ('conceptual_physics', mmlu),
        ('high_school_computer_science', mmlu),
        ('high_school_mathematics', mmlu),
        ('international_law', mmlu),
        ('high_school_statistics', mmlu),
        ('formal_logic', mmlu),
        ('high_school_psychology', mmlu),
        ('college_mathematics', mmlu),
        ('human_aging', mmlu),
        ('elementary_mathematics', mmlu),
        ('high_school_geography', mmlu),
        ('logical_fallacies', mmlu),
        ('college_chemistry', mmlu),
        ('high_school_government_and_politics', mmlu),
        ('college_medicine', mmlu),
        ('human_sexuality', mmlu),
        ('prehistory', mmlu),
        ('professional_psychology', mmlu),
        ('public_relations', mmlu),
        ('security_studies', mmlu),
        ('machine_learning', mmlu),
        ('global_facts', mmlu),
        ('professional_medicine', mmlu),
        ('nutrition', mmlu),
        ('philosophy', mmlu),
        ('professional_accounting', mmlu),
        ('high_school_biology', mmlu),
        ('philosophy', mmlu),
        ('college_computer_science', mmlu),
        ('econometrics', mmlu),
        ('high_school_microeconomics', mmlu),
        ('virology', mmlu),
        ('high_school_world_history', mmlu),
        ('high_school_physics', mmlu),
        ('high_school_us_history', mmlu),
        ('medical_genetics', mmlu),
        ('clinical_knowledge', mmlu),
        ('business_ethics', mmlu),
        ('high_school_european_history', mmlu),
        ############################ bigbench datasets ############################
        ('arithmetic', bigbench),
        ('bbq_lite_json', bigbench),
        ('cifar10_classification', bigbench),
        ('contextual_parametric_knowledge_conflicts', bigbench),
        ('color', bigbench),
        ('elementary_math_qa', bigbench),
        ('epistemic_reasoning', bigbench),
        ('fact_checker', bigbench),
        ('formal_fallacies_syllogisms_negation', bigbench),
        ('goal_step_wikihow', bigbench),
        ('hyperbaton', bigbench),
        ('logical_fallacy_detection',bigbench),
        ('mnist_ascii', bigbench),
        ('movie_dialog_same_or_different', bigbench),
        ('play_dialog_same_or_different', bigbench),
        ('real_or_fake_text', bigbench),
        ('social_iqa', bigbench),
        ('strategyqa', bigbench),
        ('timedial', bigbench),
        ('tracking_shuffled_objects', bigbench),
        ('vitaminc_fact_verification', bigbench),
        ('unit_conversion', bigbench),
        ('winowhy', bigbench),
        ############################ mrqa datasets ############################
        # Train and Val
        ('SQuAD', mrqa),
        ('SearchQA', mrqa),
        ('NaturalQuestionsShort', mrqa),
        ('HotpotQA', mrqa),
        ('NewsQA', mrqa),
        ('TriviaQA-web', mrqa),
        # Test
        ('BioASQ', mrqa_test),
        ('DROP', mrqa_test),
        ('DuoRC.ParaphraseRC', mrqa_test),
        ('RACE', mrqa_test),
        ('RelationExtraction', mrqa_test),
        ('TextbookQA', mrqa_test),
        # mrqa raw data
        ('mrqa', mrqa_raw)
    ]
)

