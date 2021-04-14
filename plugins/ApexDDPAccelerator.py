import torch
from pytorch_lightning.accelerators import GPUAccelerator
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class ApexDDPAccelerator(GPUAccelerator):
    def setup(self, trainer, model):
        if "cuda" not in str(self.root_device):
            raise MisconfigurationException(f"Device should be GPU, got {self.root_device} instead")
        self.set_nvidia_flags()
        torch.cuda.set_device(self.root_device)
        self.setup_optimizers(trainer)
        self.connect_precision_plugin(self.precision_plugin)
        self.connect_training_type_plugin(self.training_type_plugin, model)
