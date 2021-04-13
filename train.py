import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DeepSpeedPlugin

from data.RACEDataModule import RACEDataModule
from model.BertForRace import BertForRace

deepspeed_config = {
    "zero_allow_untested_optimizer": True,
    "optimizer": {
        "type": "OneBitAdam",
        "params": {
            "lr": 2e-5,
            "betas": [0.998, 0.999],
            "eps": 1e-5,
            "weight_decay": 1e-9,
            "cuda_aware": True,
        },
    },
    'scheduler': {
        "type": "WarmupLR",
        "params": {
            "last_batch_iteration": -1,
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-5,
            "warmup_num_steps": 100,
        }
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 32,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    # "amp": {
    #     "enabled": True,
    #     "opt_level": "O2",
    # },
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": 2,  # Enable Stage 2 ZeRO (Optimizer/Gradient state partitioning)
        # "cpu_offload": False,  # Enable Offloading optimizer state/calculation to the host CPU
        # "contiguous_gradients": True,  # Reduce gradient fragmentation.
        # "overlap_comm": True,  # Overlap reduce/backward operation of gradients for speed.
        # "allgather_bucket_size": 2e8,  # Number of elements to all gather at once.
        # "reduce_bucket_size": 2e8,  # Number of elements we reduce/allreduce at once.
    }
}

if __name__ == '__main__':
    tb_logger = pl_loggers.TensorBoardLogger('./result/asc01/')
    model = BertForRace(
        pretrained_model='./model/bert-large-uncased',
        learning_rate=2e-5,
        num_train_epochs=20,
        train_batch_size=4,
        train_all=Ture,
        use_longformer=False,
    )
    dm = RACEDataModule(
        model_name_or_path='./model/bert-large-uncased',
        datasets_loader='./data/RACELocalLoader.py',
        train_batch_size=4,
        max_seq_length=512,
        num_workers=8,
        num_preprocess_processes=8,
    )
    trainer = pl.Trainer(
        logger=tb_logger,
        gpus=-1 if torch.cuda.is_available() else None,
        plugins=DeepSpeedPlugin(config=deepspeed_config),
        accelerator='ddp',
        # amp_backend='apex',
        # amp_level='O2',
        precision=16,
        # gradient_clip_val=1.0,
        max_epochs=20,
        # accumulate_grad_batches=2,
    )
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
