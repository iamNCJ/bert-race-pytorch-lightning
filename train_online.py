import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers

from data.RACEDataModule import RACEDataModule
from model.BertForRace import BertForRace

if __name__ == '__main__':
    tb_logger = pl_loggers.TensorBoardLogger('result/asc01/')
    model = BertForRace(
        learning_rate=2e-5,
        num_train_epochs=10,
        train_all=False,
    )
    dm = RACEDataModule()
    trainer = pl.Trainer(
        logger=tb_logger,
        gpus=-1 if torch.cuda.is_available() else None,
        amp_backend='apex',
        amp_level='O2',
        precision=16,
        gradient_clip_val=1.0,
        max_epochs=10,
        accumulate_grad_batches=1,
    )
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
