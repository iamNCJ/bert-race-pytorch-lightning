import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from data.RACEDataModule import RACEDataModule
from model.BertForRace import BertForRace
from model.CheckptEnsemble import CheckptEnsemble

import os

ckpt_names = ["epochepoch=00-v7.ckpt", "epochepoch=01-v5.ckpt", "epochepoch=02-v4.ckpt", "epochepoch=03-v5.ckpt"]

if __name__ == '__main__':
    tb_logger = pl_loggers.TensorBoardLogger('./result/asc02/')
    # ckpt_names = os.listdir("./result/checkpoints")
    ckpt_paths = ["./result/checkpoints/" + name for name in ckpt_names]
    model = CheckptEnsemble(ckpt_paths)
    dm = RACEDataModule(
        model_name_or_path='./model/bert-large-uncased',
        datasets_loader='./data/RACELocalLoader.py',
        train_batch_size=4,
        max_seq_length=512,
        num_workers=8,
        num_preprocess_processes=48,
    )
    trainer = pl.Trainer(
        logger=tb_logger,
        gpus=-1 if torch.cuda.is_available() else None,
        amp_backend='native',
        amp_level='O2',
        precision=16,
        accelerator='ddp',
        gradient_clip_val=1.0,
        max_epochs=4,
        plugins='ddp_sharded',
        val_check_interval=0.2,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
        # accumulate_grad_batches=2,
    )
    trainer.predict(model, dm)
