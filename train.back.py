import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers

from data.RACEDataModule import RACEDataModule
from model.BertForRace import BertForRace

if __name__ == '__main__':
    tb_logger = pl_loggers.TensorBoardLogger('./result/asc01/')
    model = BertForRace(
        pretrained_model='./model/bert-large-uncased',
        learning_rate=1e-5,
        num_train_epochs=20,
        train_batch_size=4,
        train_all=False,
    )
    dm = RACEDataModule(
        model_name_or_path='./model/bert-large-uncased',
        datasets_loader='./data/RACELocalLoader.py',
        train_batch_size=4,
        max_seq_length=512,
        num_workers=8,
        num_preprocess_processes=16,
    )
    trainer = pl.Trainer(
        logger=tb_logger,
        gpus=-1 if torch.cuda.is_available() else None,
        accelerator='ddp',
	plugins='ddp_sharded',
        #amp_backend='apex',
        amp_level='O2',
        precision=16,
        gradient_clip_val=1.0,
        max_epochs=20,
        accumulate_grad_batches=2,
	#limit_train_batches=0.3,
	#limit_val_batches=0.3,
    )
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
