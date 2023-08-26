import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import wandb
import numpy as np
from pathlib import Path
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from robustbench import benchmark

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import get_original_cwd
import logging
from apex import amp
from robustarch.utils import (
    configure_optimizers,
    get_datasets,
    pad_str,
    save_checkpoint,
    make_linear_schedule,
)
from robustarch.adv_train import train, test_natural, test_pgd

from robustarch.models.model_torch import TorchModel
from robustarch.models.model import NormalizedConfigurableModel
from robustarch.models.model_cifar import NormalizedWideResNet

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="main", version_base="1.2")
def main(cfg: DictConfig):
    ### initialize parameters and folders
    device = torch.device(cfg.train_test.device)
    cwd = get_original_cwd()
    cudnn.benchmark = True

    # log all configurations
    log.info(f"\n{pad_str(' ARGUMENTS ')}\n{OmegaConf.to_yaml(cfg)}\n{pad_str('')}")

    ### Create models (configurable models not implemented)
    log.info(f"=> creating model:")
    if cfg.model.model_source == "torch":
        model = TorchModel(
            cfg.model.arch, cfg.dataset.mean, cfg.dataset.std, **cfg.model.kwargs
        )
    elif cfg.model.model_source == "local":
        model = NormalizedConfigurableModel(
            cfg.dataset.mean,
            cfg.dataset.std,
            **hydra.utils.instantiate(cfg.model.kwargs),
        )
    elif cfg.model.model_source == "local_cifar":
        model = NormalizedWideResNet(
            cfg.dataset.mean,
            cfg.dataset.std,
            **hydra.utils.instantiate(cfg.model.kwargs),
        )
    else:
        raise NotImplementedError
    model.to(device)

    ### criterion, optimizer and lr scheduler
    criterion = nn.CrossEntropyLoss().to(device)

    param_groups = configure_optimizers(model)
    optimizer_name = cfg.train_test.optim.lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            params=param_groups,
            lr=cfg.train_test.lr,
            momentum=cfg.train_test.momentum,
            weight_decay=cfg.train_test.weight_decay,
        )
    elif optimizer_name == "adamw":
        raise NotImplementedError

    ### amp half precision
    if cfg.train_test.half and not cfg.train_test.evaluate:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model = nn.DataParallel(model)

    ### Resume if training <accidentally stops/enters next phase>
    if cfg.train_test.resume:
        prev_model_dir = os.path.normpath(Path(cwd) / cfg.train_test.resume)
        if os.path.isfile(prev_model_dir):
            log.info(f"=> loading checkpoint '{prev_model_dir}'")
            ckpt = torch.load(prev_model_dir)
            start_epoch = ckpt["epoch"]
            best_prec1 = ckpt["best_prec1"]
            model.load_state_dict(ckpt["state_dict"])
            if ckpt["optimizer"]:
                optimizer.load_state_dict(ckpt["optimizer"])
    else:
        log.info(f"=> no checkpoint found at '{cfg.train_test.resume}'")
        start_epoch = cfg.train_test.start_epoch

    ### Load dataset
    train_loader, test_loader = get_datasets(cfg)

    ### Evaluate/Test
    if cfg.train_test.evaluate:
        if cfg.attack.test.name == "pgd":
            log.info(pad_str(" Performing PGD Attacks "))
            test_pgd(model, test_loader, criterion, device, cfg, log)
            test_natural(model, test_loader, criterion, device, cfg, log)
        elif cfg.attack.test.name == "aa":
            model.eval()
            test_aug = list()
            if cfg.train_test.mode == "at_pgd":
                if cfg.dataset.dataset == "imagenet":
                    test_aug.extend(
                        [
                            transforms.Resize(256),
                            transforms.CenterCrop(cfg.train_test.crop_size),
                        ]
                    )
            test_aug.extend(
                [
                    transforms.ToTensor(),
                ]
            )
            test_transform = transforms.Compose(test_aug)
            log.info(
                benchmark(
                    model,
                    dataset=cfg.dataset.dataset,
                    data_dir=cfg.dataset.data_dir,
                    device=device,
                    batch_size=cfg.attack.test.batch_size,
                    eps=cfg.attack.test.eps / 255.0,
                    preprocessing=test_transform,
                    n_examples=cfg.attack.test.n_examples,
                    threat_model="Linf"
                    if cfg.attack.test.norm == "linf"
                    else cfg.attack.test.norm,
                )
            )

        return

    # visualization
    if cfg.visualization.tool == "wandb" and cfg.visualization.entity:
        log.info(f"=> Visualization with wandb")
        wandb.init(
            project=cfg.visualization.project,
            entity=cfg.visualization.entity,
            resume=True,
        )
        wandb.run.name = cfg.name

    # trained model dir (add if)
    model_dir = Path(cwd) / cfg.train_test.model_dir
    if cfg.train_test.mode == "fat":
        model_dir = model_dir / cfg.train_test.phase
    model_dir.mkdir(parents=True, exist_ok=True)

    # lr + eps schedule
    eps_schedule = None
    if cfg.train_test.mode == "fat":
        lr_schedule = lambda t: np.interp(
            t, cfg.train_test.lr_epochs, cfg.train_test.lr_values
        )
        total_epochs = cfg.train_test.end_epoch
    else:
        total_epochs = cfg.train_test.end_epoch - cfg.train_test.start_epoch
        if cfg.train_test.schedule.lower() == "cosine":
            lr_schedule = CosineLRScheduler(
                optimizer,
                t_initial=total_epochs,
                lr_min=cfg.train_test.min_lr,
                warmup_lr_init=cfg.train_test.warmup_lr,
                warmup_t=cfg.train_test.warmup_epochs,
                cycle_mul=cfg.train_test.lr_cycle_mul,
                cycle_decay=cfg.train_test.lr_cycle_decay,
                cycle_limit=cfg.train_test.lr_cycle_limit,
            )
        elif cfg.train_test.schedule.lower() == "step":
            lr_schedule = StepLRScheduler(
                optimizer,
                decay_t=cfg.train_test.decay_t,
                decay_rate=cfg.train_test.decay_rate,
                warmup_t=cfg.train_test.warmup_epochs,
                warmup_lr_init=cfg.train_test.warmup_lr,
            )
        total_epochs = total_epochs + cfg.train_test.cooldown_epochs

        # eps schedule
        if (
            cfg.attack.train.eps_schedule
            and cfg.attack.train.eps_schedule.lower() == "linear"
            and cfg.attack.train.eps_schedule_epochs
        ):
            eps_schedule = make_linear_schedule(
                1.0 * cfg.attack.train.eps / cfg.dataset.max_color_value,
                cfg.attack.train.eps_schedule_epochs,
                cfg.attack.train.zero_eps_epochs,
            )

    ### Train
    best_prec1 = 0.0
    for epoch in range(start_epoch, total_epochs):
        # Train one epoch
        train(
            model,
            train_loader,
            optimizer,
            criterion,
            lr_schedule,
            epoch,
            device,
            cfg,
            log,
            eps_schedule,
        )

        # Test natural accuracy
        prec1 = test_natural(model, test_loader, criterion, device, cfg, log)

        if cfg.visualization.tool == "wandb" and cfg.visualization.entity:
            wandb.log({"Test natural accuracy": prec1}, step=epoch)

        # Save checkpoint based on best natural accuracy
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        model_info = dict(
            tag=cfg.name,
            best_prec1=best_prec1,
            epoch=epoch + 1,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
        )
        save_checkpoint(model_info, model_dir, is_best)


if __name__ == "__main__":
    main()
