import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
# from pytorch_lightning.callbacks import ModelCheckpoint
import logging

from data.RACEDataModuleForALBERT import RACEDataModuleForALBERT
from model.ALBERTForRace import ALBERTForRace

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    ete_start=time.time()

    train_batch_size = 4
    pretrained_model = './model/pretrained_model_asc'
    num_train_epochs = 3

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            torch.device('cuda'), 2, False, True))    

    tb_logger = pl_loggers.TensorBoardLogger('./result/asc001/')
    pl.seed_everything(42)
    model = ALBERTForRace(
        pretrained_model=pretrained_model,
        learning_rate=2e-5,
        num_train_epochs=num_train_epochs,
        train_batch_size=4,
        train_all=True,
    )

    train_start = time.time()
    dm = RACEDataModuleForALBERT(
        model_name_or_path=pretrained_model,
        datasets_loader='./data/RACELocalLoader.py',
        train_batch_size=train_batch_size,
        max_seq_length=512,
        num_workers=8,
        num_preprocess_processes=96,
        use_sentence_selection=True,
        best_k_sentences=25,
    )

    train_example = dm.dataset['train']

    num_train_steps = int(len(train_examples) / train_batch_size / 1 * num_train_epochs)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    
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
        max_epochs=num_train_epochs,
        plugins='ddp_sharded',
        val_check_interval=0.2,
        # limit_train_batches=0.1,
        # limit_val_batches=0.1,
        # accumulate_grad_batches=2,
    )
    trainer.fit(model, dm)
    finish_time = time.time()
    logger.info("ete_time: {}, training_time: {}".format(finish_time-ete_start, finish_time-train_start))

    # logger.info("***** Running evaluation: Dev *****")
    # logger.info("  Num examples = %d", len(eval_examples))
    # logger.info("  Batch size = %d", args.eval_batch_size)