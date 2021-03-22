import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers

from data.RACEDataModule import RACEDataModule
from model.BertForRace import BertForRace

if __name__ == '__main__':
    tb_logger = pl_loggers.TensorBoardLogger('result/asc01/')
    model = BertForRace(
        pretrained_model='./bert-large-uncased-vocab.txt',
        bert_config='bert-large-uncased/bert_config.json',
        learning_rate=2e-5,
    )
    dm = RACEDataModule(
        vocal_model_name_or_path='./bert-large-uncased-vocab.txt',
        datasets_loader='RACELocalLoader.py',
    )
    trainer = pl.Trainer(
        logger=tb_logger,
        gpus=-1 if torch.cuda.is_available() else None,
        amp_level='O2',
        precision=16,
        gradient_clip_val=1.0,
        max_epochs=5
    )
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)