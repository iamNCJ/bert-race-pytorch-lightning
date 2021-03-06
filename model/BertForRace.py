from typing import Any, List

import pytorch_lightning as pl
import torch
from transformers import BertConfig, BertForMultipleChoice, AdamW, get_constant_schedule_with_warmup

from data.RACEDataModule import RACEDataModule
from model.BertLongAttention import BertLongAttention


class BertForRace(pl.LightningModule):
    def __init__(
            self,
            pretrained_model: str = 'bert-large-uncased',
            learning_rate: float = 2e-5,
            gradient_accumulation_steps: int = 1,
            num_train_epochs: float = 3.0,
            train_batch_size: int = 32,
            warmup_proportion: float = 0.1,
            attention_window: int = 128,
            train_all: bool = False,
            use_bert_adam: bool = True,
            use_longformer: bool = False,
    ):
        super().__init__()
        self.config = BertConfig.from_pretrained(pretrained_model, num_choices=4)
        self.model = BertForMultipleChoice.from_pretrained(pretrained_model, config=self.config)
        # self.model.bert.dropout = nn.Dropout(0.5)

        if not train_all:
            for param in self.model.bert.parameters():
                param.requires_grad = False
            for param in self.model.bert.pooler.parameters():
                param.requires_grad = True
            # for param in self.model.bert.encoder.layer[15:24].parameters():
            #     param.requires_grad = True
            # for param in self.model.bert.encoder.layer[15].output.parameters():
            #     param.requires_grad = True

        if use_longformer:
            current_max_pos, embed_size = self.model.bert.embeddings.position_embeddings.weight.shape
            max_pos = 512
            self.config.max_position_embeddings = max_pos
            assert max_pos >= current_max_pos
            # allocate a larger position embedding matrix
            new_pos_embed = self.model.bert.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
            print(new_pos_embed.shape)
            print(self.model.bert.embeddings.position_embeddings)
            # copy position embeddings over and over to initialize the new position embeddings
            k = 0
            step = current_max_pos
            while k < max_pos - 1:
                new_pos_embed[k:(k + step)] = self.model.bert.embeddings.position_embeddings.weight
                k += step
            print(new_pos_embed.shape)
            self.model.bert.embeddings.position_embeddings.weight.data = new_pos_embed
            self.model.bert.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)
            # model.bert.embeddings.position_ids = torch.from_numpy(
            #     tf.range(new_pos_embed.shape[0], dtype=tf.int32).numpy()[tf.newaxis, :])
            # model.bert.embeddings.position_embeddings = torch.nn.Embedding.from_pretrained(new_pos_embed)

            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            self.config.attention_window = [attention_window] * self.config.num_hidden_layers
            for i, layer in enumerate(self.model.bert.encoder.layer):
                longformer_self_attn = BertLongAttention(self.config, layer_id=i)
                longformer_self_attn.query = layer.attention.self.query
                longformer_self_attn.key = layer.attention.self.key
                longformer_self_attn.value = layer.attention.self.value

                longformer_self_attn.query_global = layer.attention.self.query
                longformer_self_attn.key_global = layer.attention.self.key
                longformer_self_attn.value_global = layer.attention.self.value

                layer.attention.self = longformer_self_attn
            # self.config.attention_window = [attention_window] * self.config.num_hidden_layers
            # for i, layer in enumerate(self.model.bert.encoder.layer):
            #     # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            #     layer.attention.self = BertLongSelfAttention(self.config, layer_id=i)

        # print model layers and config
        print(self.config)
        for name, params in self.model.named_parameters():
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
        param_optimizer = list(self.model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
        ) if self.use_bert_adam else torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=self.learning_rate
        )

        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            # num_training_steps=self.total_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def forward(self, **inputs):
        return self.model(**inputs)

    def compute(self, batch):
        outputs = self.model(
            input_ids=batch['input_ids'].reshape(batch['input_ids'].shape[0], 4, -1),
            token_type_ids=batch['token_type_ids'].reshape(batch['token_type_ids'].shape[0], 4, -1),
            attention_mask=batch['attention_mask'].reshape(batch['attention_mask'].shape[0], 4, -1),
            position_ids=batch['position_ids'].reshape(batch['attention_mask'].shape[0], 4, -1),
            labels=batch['label'],
        )
        labels_hat = torch.argmax(outputs.logits, dim=1)
        correct_count = torch.sum(batch['label'] == labels_hat)
        return outputs.loss, correct_count

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
    model = BertForRace()
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
