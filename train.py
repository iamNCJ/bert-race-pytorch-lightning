import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers

from data.RACEDataModule import RACEDataModule
from model.LongformerForRace import LongformerForRace
from plugins.ApexDDP import ApexDDP

if __name__ == '__main__':
    tb_logger = pl_loggers.TensorBoardLogger('./result/asc01/')
    model = LongformerForRace(
        pretrained_model='./model/bert-large-uncased',
        learning_rate=2e-5,
        num_train_epochs=10,
        train_batch_size=8,
        train_all=False,
    )
    dm = RACEDataModule(
        model_name_or_path='./model/bert-large-uncased',
        datasets_loader='./data/RACELocalLoader.py',
        train_batch_size=32,
        max_seq_length=128,
        num_workers=8,
        num_preprocess_processes=8,
    )
    trainer = pl.Trainer(
        logger=tb_logger,
        gpus=1 if torch.cuda.is_available() else None,
        # accelerator='ddp',
        amp_backend='apex',
        amp_level='O2',
        precision=16,
        gradient_clip_val=1.0,
        max_epochs=20,
        # accumulate_grad_batches=2,
        # plugins=[ApexDDP()]
    )
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
