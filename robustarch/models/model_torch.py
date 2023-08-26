from typing import List, Any
import torchvision.models as models
import torch
import torch.nn as nn
from advertorch.utils import NormalizeByChannelMeanStd


class TorchModel(nn.Module):
    def __init__(
        self, arch_name: str, mean: List[float], std: List[float], **kwargs: Any
    ):
        super().__init__()
        # can be modified to allow pretrained model here
        self.model = models.__dict__[arch_name](**kwargs)

        assert len(mean) == len(std)
        self.normalization = NormalizeByChannelMeanStd(mean=mean, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalization(x)
        x = self.model(x)
        return x
