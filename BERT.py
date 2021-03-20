from collections import OrderedDict

import pytorch_lightning as pl
import torch
from transformers import BertConfig, BertForMultipleChoice, AdamW, get_linear_schedule_with_warmup
from RACEDataModule import RACEDataModule


class BertForRace(pl.LightningModule):
    def __init__(self,
                 pretrained_model: str = 'bert-large-uncased',
                 bert_config: str = 'bert-large-uncased',  # pretrained_model+'/bert_config.json'
                 learning_rate: float = 0.01,
                 gradient_accumulation_steps: int = 1,
                 num_train_epochs: float = 3.0,
                 train_batch_size: int = 32):
        super().__init__()
        self.config = BertConfig.from_pretrained(bert_config, num_choices=4)
        print(self.config)
        self.bert_model = BertForMultipleChoice.from_pretrained(pretrained_model, config=self.config)

        for param in self.bert_model.bert.parameters():
            param.requires_grad = False

        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size

        self.warmup_steps = 10
        self.total_steps = 0

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()

            # Calculate total steps
            self.total_steps = int(len(train_loader.dataset)
                                   / self.train_batch_size / self.gradient_accumulation_steps * self.num_train_epochs)

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.bert_model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # t_total = num_train_steps
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.learning_rate,
                          )

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def forward(self, **inputs):
        return self.bert_model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self.bert_model(
            input_ids=batch['input_ids'].reshape(batch['input_ids'].shape[0], 4, -1),
            token_type_ids=batch['token_type_ids'].reshape(batch['token_type_ids'].shape[0], 4, -1),
            attention_mask=batch['attention_mask'].reshape(batch['attention_mask'].shape[0], 4, -1),
            labels=batch['label'],
        )

        self.log('train_loss', outputs.loss)

        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self.bert_model(
            input_ids=batch['input_ids'].reshape(batch['input_ids'].shape[0], 4, -1),
            token_type_ids=batch['token_type_ids'].reshape(batch['token_type_ids'].shape[0], 4, -1),
            attention_mask=batch['attention_mask'].reshape(batch['attention_mask'].shape[0], 4, -1),
            labels=batch['label'],
        )

        self.log('val_loss', outputs.loss)

        labels_hat = torch.argmax(outputs.logits, dim=1)
        correct_count = torch.sum(batch['label'] == labels_hat)

        return outputs.loss


if __name__ == '__main__':
    model = BertForRace()
    dm = RACEDataModule()
    trainer = pl.Trainer(gpus=1, amp_level='O2', precision=16, max_epochs=10)
    trainer.fit(model, dm)
