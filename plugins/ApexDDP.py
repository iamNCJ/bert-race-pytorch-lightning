import os

import torch
from pytorch_lightning import _logger as log
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.plugins import DDPPlugin

try:
    from apex.parallel import DistributedDataParallel
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")



class ApexDDP(DDPPlugin):
    """
    trainer = Trainer(..., plugins=[ApexDDP()])
    """

    def init_ddp_connection(self, global_rank: int, world_size: int) -> None:
        os.environ["MASTER_ADDR"] = str(self.cluster_environment.master_address())
        os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())
        os.environ["WORLD_SIZE"] = str(self.cluster_environment.world_size())

        if not torch.distributed.is_initialized():
            log.info(f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
            torch.distributed.init_process_group('nccl', rank=global_rank, world_size=world_size)

    def configure_ddp(self):
        self.pre_configure_ddp()
        self._model = DistributedDataParallel(
            LightningDistributedModule(self.model),
            delay_allreduce=True,
            **self._ddp_kwargs,
        )
