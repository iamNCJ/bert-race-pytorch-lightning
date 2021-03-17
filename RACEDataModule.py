from functools import partial
from typing import Optional, Dict

import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizerFast
import datasets


class RACEDataModule(pl.LightningDataModule):
    def __init__(
            self,
            model_name_or_path: str = 'bert-large-uncased',
            task_name: str = 'all',
            max_seq_length: int = 128,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            **kwargs
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        # self.cache_dir = './'

        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name_or_path, use_fast=True)
        self.dataset = None

    def setup(self, stage: Optional[str] = None):
        self.dataset = datasets.load_dataset('race', self.task_name)
        preprocessor = partial(self.preprocess, self.tokenizer, self.max_seq_length)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                preprocessor,
                # batched=True,
                remove_columns=['example_id'],
            )
            self.dataset[split].set_format(type='torch',
                                           columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
            # self.dataset[split] = torch.utils.data.DataLoader(self.dataset[split])

    def prepare_data(self):
        datasets.load_dataset('race', self.task_name)
        BertTokenizerFast.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset['train'],
                          sampler=RandomSampler(self.dataset['train']),
                          batch_size=self.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset['validation'],
                          sampler=SequentialSampler(self.dataset['validation']),
                          batch_size=self.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'],
                          sampler=SequentialSampler(self.dataset['test']),
                          batch_size=self.eval_batch_size)

    # auto cache tokens
    @staticmethod
    def preprocess(tokenizer: BertTokenizerFast, max_seq_length: int, x: Dict) -> Dict:
        choices_features = []
        max_len = max_seq_length
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

        option: str
        for option in x["options"]:
            text_a = x["article"]
            if x["question"].find("_") != -1:
                text_b = x["question"].replace("_", option)
            else:
                text_b = x["question"] + " " + option

            inputs = tokenizer.encode_plus(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_len,
                truncation=True
            )
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
            attention_mask = [1] * len(input_ids)

            pad_token_id = tokenizer.pad_token_id
            padding_length = max_len - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_id] * padding_length)

            assert len(input_ids) == max_len, "Error with input length {} vs {}".format(len(input_ids), max_len)
            assert len(attention_mask) == max_len, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                             max_len)
            assert len(token_type_ids) == max_len, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                             max_len)

            choices_features.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            })

        labels = label_map.get(x["answer"], -1)
        label = torch.tensor(labels).long()

        return {
            "id": x["example_id"],
            "label": label,
            "input_ids": torch.tensor([cf["input_ids"] for cf in choices_features]),
            "attention_mask": torch.tensor([cf["attention_mask"] for cf in choices_features]),
            "token_type_ids": torch.tensor([cf["token_type_ids"] for cf in choices_features]),
        }


if __name__ == '__main__':
    dm = RACEDataModule()
    dm.setup('train')
