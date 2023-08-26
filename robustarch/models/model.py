from typing import Any, Callable, Dict, Optional, Tuple
from torch import Tensor
import math
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from advertorch.utils import NormalizeByChannelMeanStd
from robustarch.utils import PSiLU, PSSiLU

INPLACE_ACTIVATIONS = [nn.ReLU]
NORMALIZATIONS = [nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm, nn.LayerNorm]


class NormActivationConv(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None

        layers = list()

        if norm_layer is not None:
            layers.append(norm_layer(in_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))

        layers.append(
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        )
        super().__init__(*layers)
        self.out_channels = out_channels


class NormActivationConv2d(NormActivationConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.Conv2d,
        )


class BottleneckTransform(nn.Sequential):
    "Transformation in a Bottleneck: 1x1, kxk (k=3, 5, 7, ...) [+SE], 1x1"
    "Supported archs: [preact] [norm func+num] [act func+num] [conv kernel]"

    def __init__(
        self,
        width_in: int,
        width_out: int,
        kernel: int,
        stride: int,
        dilation: int,
        norm_layer: list[Callable[..., nn.Module]],
        activation_layer: list[Callable[..., nn.Module]],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
        se_activation: Optional[Callable[..., nn.Module]],
        ConvBlock: Callable[..., nn.Module],
    ):
        # compute transform params
        w_b = int(
            round(width_out * bottleneck_multiplier)
        )  # bottleneck_multiplier > 1 for inverted bottleneck
        g = w_b // group_width
        assert len(norm_layer) == 3
        assert len(activation_layer) == 3
        assert g > 0, f"Group convolution groups {g} should be greater than 0."
        assert (
            w_b % g == 0
        ), f"Convolution input channels {w_b} is not divisible by {g} groups."

        layers: OrderedDict[str, nn.Module] = OrderedDict()
        layers["a"] = ConvBlock(
            width_in,
            w_b,
            kernel_size=1,
            stride=1,
            norm_layer=norm_layer[0],
            activation_layer=activation_layer[0],
            inplace=True if activation_layer[0] in INPLACE_ACTIVATIONS else None,
        )

        layers["b"] = ConvBlock(
            w_b,
            w_b,
            kernel,
            stride=stride,
            groups=g,
            dilation=dilation,
            norm_layer=norm_layer[1],
            activation_layer=activation_layer[1],
            inplace=True if activation_layer[1] in INPLACE_ACTIVATIONS else None,
        )

        if se_ratio:
            assert se_activation is not None
            width_se_out = int(round(se_ratio * width_in))
            layers["se"] = SqueezeExcitation(
                input_channels=w_b,
                squeeze_channels=width_se_out,
                activation=se_activation,
            )
        if ConvBlock == Conv2dNormActivation:
            layers["c"] = ConvBlock(
                w_b,
                width_out,
                kernel_size=1,
                stride=1,
                norm_layer=norm_layer[2],
                activation_layer=None,
            )
        else:
            layers["c"] = ConvBlock(
                w_b,
                width_out,
                kernel_size=1,
                stride=1,
                norm_layer=norm_layer[2],
                activation_layer=activation_layer[2],
                inplace=True if activation_layer[2] in INPLACE_ACTIVATIONS else None,
            )

        super().__init__(layers)


class BottleneckBlock(nn.Module):
    """Bottleneck block x + F(x), where F = bottleneck transform"""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        kernel: int,
        stride: int,
        dilation: int,
        norm_layer: list[Callable[..., nn.Module]],
        activation_layer: list[Callable[..., nn.Module]],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
        se_activation: Optional[Callable[..., nn.Module]],
        ConvBlock: Callable[..., nn.Module],
        downsample_norm: Callable[..., nn.Module],
    ) -> None:
        super().__init__()

        # projection on skip connection if shape changes
        self.proj = None
        should_proj = (width_in != width_out) or (stride != 1)
        if should_proj:
            if ConvBlock == Conv2dNormActivation:
                self.proj = ConvBlock(
                    width_in,
                    width_out,
                    kernel_size=1,
                    stride=stride,
                    norm_layer=downsample_norm,
                    activation_layer=None,
                )
            elif ConvBlock == NormActivationConv2d:
                self.proj = ConvBlock(
                    width_in,
                    width_out,
                    kernel_size=1,
                    stride=stride,
                    norm_layer=None,
                    activation_layer=None,
                    bias=False,
                )

        self.F = BottleneckTransform(
            width_in,
            width_out,
            kernel,
            stride,
            dilation,
            norm_layer,
            activation_layer,
            group_width,
            bottleneck_multiplier,
            se_ratio,
            se_activation,
            ConvBlock,
        )

        if ConvBlock == Conv2dNormActivation:
            if activation_layer[2] is not None:
                if activation_layer[2] in INPLACE_ACTIVATIONS:
                    self.last_activation = activation_layer[2](inplace=True)
                else:
                    self.last_activation = activation_layer[2]()
        else:
            self.last_activation = None

    def forward(self, x: Tensor) -> Tensor:
        if self.proj is not None:
            x = self.proj(x) + self.F(x)
        else:
            x = x + self.F(x)

        if self.last_activation is not None:
            return self.last_activation(x)
        else:
            return x


class Stage(nn.Sequential):
    """Stage is a sequence of blocks with the same output shape. Downsampling block is the first in each stage"""

    """Options: stage numbers, stage depth, dense connection"""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        kernel: int,
        stride: int,
        dilation: int,
        norm_layer: list[Callable[..., nn.Module]],
        activation_layer: list[Callable[..., nn.Module]],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
        se_activation: Optional[Callable[..., nn.Module]],
        ConvBlock: Callable[..., nn.Module],
        downsample_norm: Callable[..., nn.Module],
        depth: int,
        dense_ratio: Optional[float],
        block_constructor: Callable[..., nn.Module] = BottleneckBlock,
        stage_index: int = 0,
    ):
        super().__init__()
        self.dense_ratio = dense_ratio
        for i in range(depth):
            block = block_constructor(
                width_in if i == 0 else width_out,
                width_out,
                kernel,
                stride if i == 0 else 1,
                dilation,
                norm_layer,
                activation_layer,
                group_width,
                bottleneck_multiplier,
                se_ratio,
                se_activation,
                ConvBlock,
                downsample_norm,
            )

            self.add_module(f"stage{stage_index}-block{i}", block)

    def forward(self, x: Tensor) -> Tensor:
        if self.dense_ratio:
            assert self.dense_ratio > 0
            features = list([x])
            for i, module in enumerate(self):
                input = features[-1]
                if i > 2:
                    for j in range(self.dense_ratio):
                        if j + 4 > len(features):
                            break
                        input = input + features[-3 - j]
                x = module(input)
                features.append(x)

            # output of each stage is also densely connected
            x = features[-1]
            for k in range(self.dense_ratio):
                if k + 4 > len(features):
                    break
                x = x + features[-3 - k]
        else:
            for module in self:
                x = module(x)
        return x


class Stem(nn.Module):
    """Stem for ImageNet: kxk, BN, ReLU[, MaxPool]"""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        kernel_size: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        downsample_factor: int,
        patch_size: Optional[int],
    ) -> None:
        super().__init__()

        assert downsample_factor % 2 == 0 and downsample_factor >= 2
        layers: OrderedDict[str, nn.Module] = OrderedDict()

        stride = 2
        if patch_size:
            kernel_size = patch_size
            stride = patch_size

        layers["stem"] = Conv2dNormActivation(
            width_in,
            width_out,
            kernel_size=kernel_size,
            stride=stride,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        if not patch_size and downsample_factor // 2 > 1:
            layers["stem_downsample"] = nn.MaxPool2d(
                kernel_size=3, stride=downsample_factor // 2, padding=1
            )

        self.stem = nn.Sequential(layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.stem(x)


class ConfigurableModel(nn.Module):
    def __init__(
        self,
        stage_widths: list[int],  # output width of each stage
        kernel: int,  # kernel for non-pointwise conv
        strides: list[int],  # stride in each stage
        dilation: int,  # dilation for non-pointwise conv
        norm_layer: list[
            Callable[..., nn.Module]
        ],  # norm layer in each block, length 3 for bottleneck
        activation_layer: list[
            Callable[..., nn.Module]
        ],  # activation layer in each block, length 3 for bottleneck
        group_widths: list[
            int
        ],  # group conv width in each stage, groups = width_out * bottleneck_multiplier // group_width
        bottleneck_multipliers: list[
            float
        ],  # bottleneck_multiplier > 1 for inverted bottleneck
        downsample_norm: Callable[
            ..., nn.Module
        ],  # norm layer in downsampling shortcut
        depths: list[int],  # depth in each stage
        dense_ratio: Optional[float],  # dense connection ratio
        stem_type: Callable[..., nn.Module],  # stem stage
        stem_width: int,  # stem stage output width
        stem_kernel: int,  # stem stage kernel size
        stem_downsample_factor: int,  # downscale factor in the stem stage, if > 2, a maxpool layer is added
        stem_patch_size: Optional[int],  # patchify stem patch size
        block_constructor: Callable[
            ..., nn.Module
        ] = BottleneckBlock,  # block type in body stage
        ConvBlock: Callable[
            ..., nn.Module
        ] = Conv2dNormActivation,  # block with different "conv-norm-act" order
        se_ratio: Optional[float] = None,  # squeeze and excitation (SE) ratio
        se_activation: Optional[
            Callable[..., nn.Module]
        ] = None,  # activation layer in SE block
        weight_init_type: str = "resnet",  # initialization type
        num_classes: int = 1000,  # num of classification classes
    ) -> None:
        super().__init__()

        num_stages = len(stage_widths)
        assert len(strides) == num_stages
        assert len(bottleneck_multipliers) == num_stages
        assert len(group_widths) == num_stages
        assert len(norm_layer) == len(activation_layer)
        assert (
            sum([i % 8 for i in stage_widths]) == 0
        ), f"Stage width {stage_widths} non-divisible by 8"

        # stem
        self.stem = stem_type(
            width_in=3,
            width_out=stem_width,
            kernel_size=stem_kernel,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.ReLU,
            downsample_factor=stem_downsample_factor,
            patch_size=stem_patch_size,
        )

        # stages
        current_width = stem_width
        stages = list()
        for i, (
            width_out,
            stride,
            group_width,
            bottleneck_multiplier,
            depth,
        ) in enumerate(
            zip(stage_widths, strides, group_widths, bottleneck_multipliers, depths)
        ):
            stages.append(
                (
                    f"stage{i + 1}",
                    Stage(
                        current_width,
                        width_out,
                        kernel,
                        stride,
                        dilation,
                        norm_layer,
                        activation_layer,
                        group_width,
                        bottleneck_multiplier,
                        se_ratio,
                        se_activation,
                        ConvBlock,
                        downsample_norm,
                        depth,
                        dense_ratio,
                        block_constructor,
                        stage_index=i + 1,
                    ),
                )
            )

            current_width = width_out

        self.stages = nn.Sequential(OrderedDict(stages))

        # classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(current_width, num_classes)

        # initialization
        if weight_init_type == "resnet":
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
                    # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif m in NORMALIZATIONS:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(m.bias)
        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stages(x)

        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x


class NormalizedConfigurableModel(ConfigurableModel):
    def __init__(self, mean: list[float], std: list[float], **kwargs: Any):
        super().__init__(**kwargs)

        assert len(mean) == len(std)
        self.normalization = NormalizeByChannelMeanStd(mean=mean, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalization(x)

        x = self.stem(x)
        x = self.stages(x)

        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x
