from typing import Any, List

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertConfig, BertModel, AdamW, get_linear_schedule_with_warmup

from data.RACEDataModule import RACEDataModule
from model.BertPooler import BertPooler
from model.FuseNet import FuseNet
from model.SSingleMatchNet import SSingleMatchNet


class DCMNForRace(pl.LightningModule):
    def __init__(
            self,
            pretrained_model: str = 'bert-large-uncased',
            num_choices: int = 4,
            learning_rate: float = 2e-5,
            gradient_accumulation_steps: int = 1,
            num_train_epochs: float = 3.0,
            train_batch_size: int = 32,
            warmup_proportion: float = 0.1,
            train_all: bool = False,
            use_bert_adam: bool = True,
    ):
        super().__init__()
        self.config = BertConfig.from_pretrained(pretrained_model, num_choices=4)
        self.bert = BertModel.from_pretrained(pretrained_model, config=self.config)
        self.num_choices = num_choices
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(3 * self.config.hidden_size, 1)
        self.ssmatch = SSingleMatchNet(self.config)
        self.pooler = BertPooler(self.config)
        self.fuse = FuseNet(self.config)

        if not train_all:
            for param in self.bert.parameters():
                param.requires_grad = False
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
            # for param in self.bert.encoder.layer[15:24].parameters():
            #     param.requires_grad = True
            # for param in self.bert.encoder.layer[15].output.parameters():
            #     param.requires_grad = True

        # print model layers and config
        print(self.config)
        for name, params in self.named_parameters():
            print('-->name:', name, '-->grad_require:', params.requires_grad)

        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size
        self.warmup_proportion = warmup_proportion
        self.use_bert_adam = use_bert_adam

        self.warmup_steps = 0
        self.total_steps = 0

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()

            # Calculate total steps
            self.total_steps = int(len(train_loader.dataset)
                                   / self.train_batch_size / self.gradient_accumulation_steps * self.num_train_epochs)
            self.warmup_steps = int(self.total_steps * self.warmup_proportion)

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        # classifer_param_optimizer = list(self.classifier.named_parameters())
        # ssmatch_param_optimizer = list(self.ssmatch.named_parameters())
        # fuse_param_optimizer = list(self.fuse.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            # {'params': [p for n, p in classifer_param_optimizer]},
            # {'params': [p for n, p in ssmatch_param_optimizer]},
            # {'params': [p for n, p in fuse_param_optimizer]},
            # {'params': list(self.pooler.named_parameters()), 'lr': 1e-4},
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
        ) if self.use_bert_adam else torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=self.learning_rate
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

    @staticmethod
    def seperate_seq(sequence_output, doc_len, ques_len, option_len):
        doc_seq_output = sequence_output.new(sequence_output.size()).zero_()
        doc_ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
        ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
        ques_option_seq_output = sequence_output.new(sequence_output.size()).zero_()
        option_seq_output = sequence_output.new(sequence_output.size()).zero_()
        for i in range(doc_len.size(0)):
            doc_seq_output[i, :doc_len[i]] = sequence_output[i, 1:doc_len[i] + 1]
            doc_ques_seq_output[i, :doc_len[i] + ques_len[i]] = sequence_output[i, :doc_len[i] + ques_len[i]]
            ques_seq_output[i, :ques_len[i]] = sequence_output[i, doc_len[i] + 2:doc_len[i] + ques_len[i] + 2]
            ques_option_seq_output[i, :ques_len[i] + option_len[i]] = sequence_output[i,
                                                                      doc_len[i] + 1:doc_len[i] + ques_len[i] +
                                                                                     option_len[i] + 1]
            option_seq_output[i, :option_len[i]] = sequence_output[i,
                                                   doc_len[i] + ques_len[i] + 2:doc_len[i] + ques_len[i] + option_len[
                                                       i] + 2]
        return doc_ques_seq_output, ques_option_seq_output, doc_seq_output, ques_seq_output, option_seq_output

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, doc_len=None, ques_len=None,
                option_len=None, labels=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        doc_len = doc_len.view(-1, doc_len.size(0) * doc_len.size(1)).squeeze()
        ques_len = ques_len.view(-1, ques_len.size(0) * ques_len.size(1)).squeeze()
        option_len = option_len.view(-1, option_len.size(0) * option_len.size(1)).squeeze()

        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))

        outputs = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask)
        sequence_output = outputs.last_hidden_state

        _, _, doc_seq_output, ques_seq_output, option_seq_output = self.seperate_seq(sequence_output, doc_len, ques_len, option_len)

        pa_output = self.ssmatch([doc_seq_output, option_seq_output, option_len + 1])
        ap_output = self.ssmatch([option_seq_output, doc_seq_output, doc_len + 1])
        pq_output = self.ssmatch([doc_seq_output, ques_seq_output, ques_len + 1])
        qp_output = self.ssmatch([ques_seq_output, doc_seq_output, doc_len + 1])
        qa_output = self.ssmatch([ques_seq_output, option_seq_output, option_len + 1])
        aq_output = self.ssmatch([option_seq_output, ques_seq_output, ques_len + 1])

        pa_output_pool, _ = pa_output.max(1)
        ap_output_pool, _ = ap_output.max(1)
        pq_output_pool, _ = pq_output.max(1)
        qp_output_pool, _ = qp_output.max(1)
        qa_output_pool, _ = qa_output.max(1)
        aq_output_pool, _ = aq_output.max(1)

        pa_fuse = self.fuse([pa_output_pool, ap_output_pool])
        pq_fuse = self.fuse([pq_output_pool, qp_output_pool])
        qa_fuse = self.fuse([qa_output_pool, aq_output_pool])

        cat_pool = torch.cat([pa_fuse, pq_fuse, qa_fuse], 1)
        output_pool = self.dropout(cat_pool)
        match_logits = self.classifier(output_pool)
        match_reshaped_logits = match_logits.view(-1, self.num_choices)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            match_loss = loss_fct(match_reshaped_logits, labels)
            return match_loss, match_reshaped_logits
        else:
            return match_reshaped_logits

    def compute(self, batch):
        outputs = self(
            input_ids=batch['input_ids'].reshape(batch['input_ids'].shape[0], 4, -1),
            token_type_ids=batch['token_type_ids'].reshape(batch['token_type_ids'].shape[0], 4, -1),
            attention_mask=batch['attention_mask'].reshape(batch['attention_mask'].shape[0], 4, -1),
            doc_len=batch['article_len'],
            ques_len=batch['question_len'],
            option_len=batch['option_len'],
            labels=batch['label'],
        )
        labels_hat = torch.argmax(outputs[1], dim=1)
        correct_count = torch.sum(batch['label'] == labels_hat)
        return outputs[0], correct_count

    def training_step(self, batch, batch_idx):
        loss, correct_count = self.compute(batch)
        self.log('train_loss', loss)
        self.log('train_acc', correct_count.float() / len(batch['label']))

        return loss

    def validation_step(self, batch, batch_idx):
        loss, correct_count = self.compute(batch)

        return {
            "val_loss": loss,
            "correct_count": correct_count,
            "batch_size": len(batch['label'])
        }

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        val_acc = sum([out["correct_count"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs)
        val_loss = sum([out["val_loss"] for out in outputs]) / len(outputs)
        self.log('val_acc', val_acc)
        self.log('val_loss', val_loss)

    def test_step(self, batch, batch_idx):
        loss, correct_count = self.compute(batch)

        return {
            "test_loss": loss,
            "correct_count": correct_count,
            "batch_size": len(batch['label'])
        }

    def test_epoch_end(self, outputs: List[Any]) -> None:
        test_acc = sum([out["correct_count"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs)
        test_loss = sum([out["test_loss"] for out in outputs]) / len(outputs)
        self.log('test_acc', test_acc)
        self.log('test_loss', test_loss)


if __name__ == '__main__':
    model = DCMNForRace()
    dm = RACEDataModule()
    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.is_available() else None,
        # amp_level='O2',
        # precision=16,
        gradient_clip_val=1.0,
        max_epochs=5
    )
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
