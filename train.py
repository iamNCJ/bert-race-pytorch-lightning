import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
# from pytorch_lightning.callbacks import ModelCheckpoint

from data.RACEDataModule import RACEDataModule
from model.DCMNForRace import DCMNForRace


if __name__ == '__main__':
    tb_logger = pl_loggers.TensorBoardLogger('./result/asc01/')
    model = DCMNForRace(
        pretrained_model='./model/bert-large-uncased',
        learning_rate=2e-5,
        num_train_epochs=6,
        train_batch_size=4,
        train_all=True,
    )
    dm = RACEDataModule(
        model_name_or_path='./model/bert-large-uncased',
        datasets_loader='./data/RACELocalLoader.py',
        train_batch_size=4,
        max_seq_length=512,
        num_workers=8,
        num_preprocess_processes=48,
        use_sentence_selection=False,
    )
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath='./result/checkpoints/',
    #     filename='epoch{epoch:02d}',
    #     save_top_k=-1,
    # )
    trainer = pl.Trainer(
        logger=tb_logger,
        gpus=-1 if torch.cuda.is_available() else None,
        # callbacks=[checkpoint_callback],
        amp_backend='native',
        amp_level='O2',
        precision=16,
        accelerator='ddp',
        gradient_clip_val=1.0,
        max_epochs=6,
        plugins='ddp_sharded',
        val_check_interval=0.2,
        # limit_train_batches=0.1,
        # limit_val_batches=0.1,
        # accumulate_grad_batches=2,
    )
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
