import pytorch_lightning as pl
from transformers import BertTokenizer


class RACEDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", do_lower_case=True)


if __name__ == '__main__':
    RACEDataModule()