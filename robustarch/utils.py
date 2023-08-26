from typing import List, Dict, Tuple, Callable
import torch
import os
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import shutil

IMAGENET_PCA = {
    "eigval": torch.Tensor([0.2175, 0.0188, 0.0045]),
    "eigvec": torch.Tensor(
        [
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ]
    ),
}


class Lighting(object):
    """
    Lighting noise (see https://git.io/fhBOc)
    https://github.com/MadryLab/robustness/blob/a9541241defd9972e9334bfcdb804f6aefe24dc7/robustness/data_augmentation.py#L18
    """

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = (
            self.eigvec.type_as(img)
            .clone()
            .mul(alpha.view(1, 3).expand(3, 3))
            .mul(self.eigval.view(1, 3).expand(3, 3))
            .sum(1)
            .squeeze()
        )

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class PSiLU(nn.Module):
    def __init__(self, alpha: float = 1.0, device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.alpha = nn.Parameter(torch.empty(1, **factory_kwargs).fill_(alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.alpha * x)


class PSSiLU(nn.Module):
    def __init__(
        self, alpha: float = 1.0, beta: float = 1e-4, device=None, dtype=None
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.alpha = nn.Parameter(torch.empty(1, **factory_kwargs).fill_(alpha))
        self.beta = nn.Parameter(torch.empty(1, **factory_kwargs).fill_(beta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.beta.data = torch.clamp(self.beta.data, 0.0, 1.0)
        return (
            x * (torch.sigmoid(self.alpha * x) - self.beta) / (1.0 - self.beta + 1e-6)
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: tuple = (1, 5)
) -> List[float]:
    """Computes the accuracy over the k top predictions for the specified values of k"""

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


def configure_optimizers(model: nn.Module) -> List[Dict]:
    """
    all normalization params and all bias shouldn't be weight decayed
    code adapted from https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136
    """
    decay = set()
    no_decay = set()
    weight_decay_blacklist = (
        nn.BatchNorm2d,
        nn.LayerNorm,
        nn.InstanceNorm2d,
        nn.PReLU,
        PSiLU,
        PSSiLU,
    )
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            full_name = f"{mn}.{pn}" if mn else pn
            if pn.endswith("bias"):
                # all biases are not decayed
                no_decay.add(full_name)
            elif pn.endswith("weight") and isinstance(m, weight_decay_blacklist):
                # weights of blacklist are not decayed
                no_decay.add(full_name)
            else:
                decay.add(full_name)

    # assert all parameters are considered
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert (
        len(param_dict.keys() - union_params) == 0
    ), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

    optim_group = list(
        [
            {"params": [param_dict[fpn] for fpn in sorted(list(decay))]},
            {
                "params": [param_dict[fpn] for fpn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
    )

    return optim_group


def get_cifar10(
    root_dir: str, batch_size: int, workers: int, crop_size: int
) -> List[DataLoader]:
    """
    The transforms follow the robustness library: https://github.com/MadryLab/robustness/blob/a9541241defd9972e9334bfcdb804f6aefe24dc7/robustness/data_augmentation.py#L68
    """
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(crop_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.25, 0.25, 0.25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=root_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=root_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def get_imagenet(
    data_dir: str,
    batch_size: int,
    workers: int,
    crop_size: int,
    color_jitter: float = 0.0,
    use_lighting: bool = False,
    mode: str = "fat",
) -> List[DataLoader]:
    train_aug = list(
        [
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
        ]
    )

    if color_jitter > 0:
        cj = (float(color_jitter),) * 3
        train_aug.append(transforms.ColorJitter(*cj))

    train_aug.append(transforms.ToTensor())

    if use_lighting:
        train_aug.append(Lighting(0.05, IMAGENET_PCA["eigval"], IMAGENET_PCA["eigvec"]))

    train_transform = transforms.Compose(train_aug)

    test_aug = list()

    # Test Transform is exactly the same as `robustness` library
    if mode == "at_pgd":
        test_aug.append(transforms.Resize(256))
    test_aug.extend(
        [
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(test_aug)

    train_path = os.path.join(data_dir, "train")
    test_path = os.path.join(data_dir, "val")

    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def get_datasets(cfg):
    if cfg.dataset.dataset == "imagenet":
        return get_imagenet(
            data_dir=cfg.dataset.data_dir,
            batch_size=cfg.train_test.batch_size,
            workers=cfg.train_test.workers,
            crop_size=cfg.train_test.crop_size,
            color_jitter=cfg.train_test.color_jitter,
            use_lighting=cfg.train_test.lighting,
            mode=cfg.train_test.mode,
        )
    elif cfg.dataset.dataset == "cifar10":
        return get_cifar10(
            root_dir=cfg.dataset.data_dir,
            batch_size=cfg.train_test.batch_size,
            workers=cfg.train_test.workers,
            crop_size=cfg.train_test.crop_size,
        )


def pad_str(msg: str, total_len: int = 80) -> str:
    rem_len = total_len - len(msg)
    return f"{'*' * (rem_len // 2)}{msg}{'*' * (rem_len // 2)}"


def save_checkpoint(model_info: dict, filepath: Path, is_best: bool):
    filename = filepath / f"ckpt_epoch{model_info['epoch']}.pt"
    torch.save(model_info, filename)
    if is_best:
        shutil.copyfile(filename, filepath / "model_best.pt")


def compute_total_parameters(model: nn.Module) -> float:
    # return # of parameters in million
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    return pytorch_total_params / 1e6


# linear scheduling for eps
def make_linear_schedule(
    final: float, warmup: int, zero_eps_epochs: int
) -> Callable[[int], float]:
    def linear_schedule(step: int) -> float:
        if step < zero_eps_epochs:
            return 0.0
        if step < warmup:
            return (step - zero_eps_epochs) / warmup * final
        return final

    return linear_schedule
