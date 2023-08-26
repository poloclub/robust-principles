TORCHMODEL = ++model.arch=$(ARCH)

# torchvision model lists
RESNET18_TORCH = resnet18
RESNET50_TORCH = resnet50
RESNET101_TORCH = resnet101
RESNET152_TORCH = resnet152
WIDE_RESNET50_2_TORCH = wide_resnet50_2
WIDE_RESNET101_2_TORCH = wide_resnet101_2

# local model list
RESNET50 = model=model

SE_0_25_RELU = ++model.kwargs.se_ratio=0.25 ++model.kwargs.se_activation._target_=hydra.utils.get_class ++model.kwargs.se_activation.path=torch.nn.ReLU
STEM_WIDTH_96 = ++model.kwargs.stem_width=96
MOVE_DOWN_DOWNSAMPLE_1 = ++model.kwargs.stem_downsample_factor=2 ++model.kwargs.strides.0=2
ACT_SILU_SILU_SILU = ++model.kwargs.activation_layer.0.path=torch.nn.SiLU ++model.kwargs.activation_layer.1.path=torch.nn.SiLU ++model.kwargs.activation_layer.2.path=torch.nn.SiLU

RaResNet50 = ++model.kwargs.depths=[5,8,13,1] ++model.kwargs.stage_widths=[288,576,1120,2160] ++model.kwargs.group_widths=[36,72,140,270] $(MOVE_DOWN_DOWNSAMPLE_1) $(STEM_WIDTH_96) $(SE_0_25_RELU) $(ACT_SILU_SILU_SILU) ++model.kwargs.norm_layer.0.path=torch.nn.Identity
RaResNet101 = ++model.kwargs.depths=[7,11,18,1] ++model.kwargs.stage_widths=[336,672,1328,2624] ++model.kwargs.group_widths=[42,84,166,328] $(MOVE_DOWN_DOWNSAMPLE_1) $(STEM_WIDTH_96) $(SE_0_25_RELU) $(ACT_SILU_SILU_SILU) ++model.kwargs.norm_layer.0.path=torch.nn.Identity
RaWRN101_2 = ++model.kwargs.depths=[7,11,18,1] ++model.kwargs.stage_widths=[512,1024,2016,4032] ++model.kwargs.group_widths=[64,128,252,504] $(MOVE_DOWN_DOWNSAMPLE_1) $(STEM_WIDTH_96) $(SE_0_25_RELU) $(ACT_SILU_SILU_SILU) ++model.kwargs.norm_layer.0.path=torch.nn.Identity
RaWRN70_16 = model=cifar_model