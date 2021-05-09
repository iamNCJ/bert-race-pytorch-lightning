from functools import partial
from typing import Optional, Dict

import datasets
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizerFast
from rouge_score import rouge_scorer


class RACEDataModuleForTinyChoice(pl.LightningDataModule):
    def __init__(
            self,
            model_name_or_path: str = 'bert-large-uncased',
            datasets_loader: str = 'race',
            task_name: str = 'all',
            max_seq_length: int = 640,
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

        preprocessor = partial(self.preprocess, self.tokenizer, self.max_seq_length, )

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                preprocessor,
                # batched=True,
                remove_columns=['example_id'],
                num_proc=self.num_preprocess_processes,
                keep_in_memory=True,
            )
            self.dataset[split].set_format(type='torch',
                                           columns=['input_ids', 'token_type_ids', 'attention_mask', 'position_ids',
                                                    'label', 'indices'])

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
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        all_input_ids = np.zeros(640)
        attention_mask = np.zeros(640)
        qa_input_ids = tokenizer(f'[SEP] {x["question"]}' + ' [SEP] [CLS] '.join([''] + x['options']) + ' [SEP]',
                                 add_special_tokens=False)['input_ids']
        article_input_ids = tokenizer(x['article'], truncation=True, max_length=min(511, 640 - len(qa_input_ids)),
                                      add_special_tokens=False)['input_ids']
        all_input_ids[0:len(article_input_ids)] = article_input_ids
        all_input_ids[len(article_input_ids):len(article_input_ids) + len(qa_input_ids)] = qa_input_ids
        attention_mask[0:len(article_input_ids) + len(qa_input_ids)] = 1
        sep_indices = np.where(np.array(all_input_ids) == 102)[0]
        cls_indices = np.where(np.array(all_input_ids) == 101)[0]
        assert len(cls_indices) == 4
        relative_position_ids = np.arange(512)
        position_ids = np.empty((len(all_input_ids)), dtype=int)
        token_type_ids = np.ones((len(all_input_ids)), dtype=int)
        token_type_ids[0:sep_indices[0] + 1] = 0
        position_ids[0:sep_indices[0] + 1] = relative_position_ids[0:sep_indices[0] + 1]
        for i in range(5):
            position_ids[sep_indices[i] + 1:sep_indices[i + 1] + 1] = relative_position_ids[0:sep_indices[i + 1] - sep_indices[i]]

        labels = label_map.get(x["answer"], -1)
        label = torch.tensor(labels).long()

        return {
            "label": label,
            "input_ids": torch.tensor(all_input_ids).long(),
            "attention_mask": torch.tensor(attention_mask).long(),
            "token_type_ids": torch.tensor(token_type_ids).long(),
            "position_ids": torch.tensor(position_ids).long(),
            "indices": torch.tensor(cls_indices).long(),
        }


if __name__ == '__main__':
    dm = RACEDataModuleForTinyChoice(train_batch_size=32)
    dm.setup('train')
    d = (next(iter(dm.test_dataloader())))
    print(d)
