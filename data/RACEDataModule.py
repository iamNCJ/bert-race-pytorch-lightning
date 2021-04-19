from functools import partial
from typing import Optional, Dict

import datasets
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizerFast


class RACEDataModule(pl.LightningDataModule):
    def __init__(
            self,
            model_name_or_path: str = 'bert-large-uncased',  # './vocab.txt'
            datasets_loader: str = 'race',  # 'RACELocalLoader.py'
            task_name: str = 'all',
            max_seq_length: int = 512,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            num_workers: int = 8,
            num_preprocess_processes: int = 8,
            **kwargs
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.dataset_loader = datasets_loader
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.num_preprocess_processes = num_preprocess_processes

        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name_or_path, use_fast=True, do_lower_case=True)
        self.dataset = None

    def setup(self, stage: Optional[str] = None):
        self.dataset = datasets.load_dataset(self.dataset_loader, self.task_name)

        preprocessor = partial(self.preprocess, self.tokenizer, self.max_seq_length)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                preprocessor,
                # batched=True,
                remove_columns=['example_id'],
                num_proc=self.num_preprocess_processes,
            )
            self.dataset[split].set_format(type='torch',
                                           columns=['input_ids', 'token_type_ids', 'attention_mask', 'article_len',
                                                    'question_len', 'option_len', 'label'])

    def prepare_data(self):
        datasets.load_dataset(self.dataset_loader, self.task_name)
        BertTokenizerFast.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset['train'],
                          sampler=RandomSampler(self.dataset['train']),
                          batch_size=self.train_batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset['validation'],
                          sampler=SequentialSampler(self.dataset['validation']),
                          batch_size=self.eval_batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'],
                          sampler=SequentialSampler(self.dataset['test']),
                          batch_size=self.eval_batch_size,
                          num_workers=self.num_workers)

    # auto cache tokens
    @staticmethod
    def preprocess(tokenizer: BertTokenizerFast, max_seq_length: int, x: Dict) -> Dict:
        choices_features = []
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        question = x["question"]
        question_len = len(tokenizer.tokenize(question))

        option: str
        for option in x["options"]:
            question_option = question + option
            # if question.find("_") != -1:
            #     # fill in the banks questions
            #     question_option = question.replace("_", option)
            # else:
            #     question_option = question + " [SEP] " + option
            option_len = len(tokenizer.tokenize(option))

            inputs = tokenizer(
                x["article"],
                question_option,
                add_special_tokens=True,
                max_length=max_seq_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            token_type_ids = np.array(inputs['token_type_ids'])
            inputs['article_len'] = int(np.where(token_type_ids == 1)[1][0]) - 2
            # inputs['question_len'] = question_len
            inputs['option_len'] = option_len

            choices_features.append(inputs)

        labels = label_map.get(x["answer"], -1)
        label = torch.LongTensor(labels)

        return {
            "label": label,
            "input_ids": torch.cat([cf["input_ids"] for cf in choices_features]).reshape(-1),
            "attention_mask": torch.cat([cf["attention_mask"] for cf in choices_features]).reshape(-1),
            "token_type_ids": torch.cat([cf["token_type_ids"] for cf in choices_features]).reshape(-1),
            "article_len": torch.LongTensor([cf["article_len"] for cf in choices_features]),
            "question_len": torch.LongTensor([question_len] * 4),
            # "question_len": torch.Tensor([cf["question_len"] for cf in choices_features]),
            "option_len": torch.LongTensor([cf["option_len"] for cf in choices_features]),
        }


if __name__ == '__main__':
    dm = RACEDataModule(train_batch_size=32)
    dm.setup('train')
    d = (next(iter(dm.test_dataloader())))
    print(d)
