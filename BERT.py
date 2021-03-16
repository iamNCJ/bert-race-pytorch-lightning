import pytorch_lightning as pl
from transformers import AutoConfig, BertForMultipleChoice, AdamW, get_linear_schedule_with_warmup


class BertForRace(pl.LightningModule):
    def __init__(self, pretrained_model, learning_rate=0.01):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model, num_choices=4)
        print(self.config)
        self.bert_model = BertForMultipleChoice.from_pretrained(pretrained_model, config=self.config)
        # print(self.bert_model.named_parameters())
        # param_optimizer = list(self.bert_model.named_parameters())
        # print(len(param_optimizer))
        self.learning_rate = learning_rate

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        # param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        # print(len(param_optimizer))

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
                          # warmup=args.warmup_proportion,
                          # t_total=t_total
                          )

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]
        # return optimizer


if __name__ == '__main__':
    model = BertForRace(pretrained_model="bert-large-uncased")
