import pytorch_lightning as pl
from transformers import AutoConfig, BertForMultipleChoice


class BertForRace(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained("bert-large-uncased", num_choices=4)
        print(self.config)
        self.bert_model = BertForMultipleChoice.from_pretrained("bert-large-uncased", config=self.config)
        print(self.bert_model.named_parameters())


if __name__ == '__main__':
    model = BertForRace()
