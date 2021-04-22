from typing import Any, List

import pytorch_lightning as pl
import torch
from transformers import BertConfig, BertForMultipleChoice, LongformerSelfAttention, AdamW, get_linear_schedule_with_warmup

from data.RACEDataModule import RACEDataModule
from model.BertLongAttention import BertLongAttention

import BertForRace

class CheckptEnsemble(pl.LightningModule):
    def __init__(self, checkpoints: List[str]):
        self.models = [BertForRace.load_from_checkpoint(ckpt) for ckpt in checkpoints]

    def forward(self, **inputs):
        outputs = [model(inputs) for model in self.models]
        return outputs
    
    def predict_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'].reshape(batch['input_ids'].shape[0], 4, -1),
            token_type_ids=batch['token_type_ids'].reshape(batch['token_type_ids'].shape[0], 4, -1),
            attention_mask=batch['attention_mask'].reshape(batch['attention_mask'].shape[0], 4, -1),
            labels=batch['label']
        )
        logitses = [output.logits for output in outputs]
        labels_hat = torch.stack(logitses, dim=0).sum(dim=0).argmax(dim=1)
        correct_count = torch.sum(batch['label'] == labels_hat)
        self.log('predict_acc', correct_count.float() / len(batch['label']))

