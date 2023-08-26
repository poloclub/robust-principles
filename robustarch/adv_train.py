from typing import Any, Callable
import torch
from torch.distributions import Uniform
import time
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig
import logging
import wandb
from apex import amp
from advertorch.attacks import LinfPGDAttack

from robustarch.utils import AverageMeter, accuracy, pad_str


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    lr_schedule: Any,
    epoch: int,
    device: torch.device,
    cfg: DictConfig,
    log: logging.Logger,
    eps_schedule: Callable = None,
):
    """
    Normalization step belongs to the model. It should be the first step of forward function.
    """

    # initialize all meters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()  # train mode
    mark_time = time.time()
    train_eps = 1.0 * cfg.attack.train.eps / cfg.dataset.max_color_value
    test_eps = 1.0 * cfg.attack.test.eps / cfg.dataset.max_color_value
    assert train_eps <= 1.0
    assert test_eps <= 1.0

    if cfg.train_test.mode.lower() == "fat":
        sampler = Uniform(low=-test_eps, high=test_eps)
    else:
        if eps_schedule:
            log.info(pad_str(f" Train with eps={eps_schedule(epoch)} "))
        else:
            log.info(pad_str(f" Train with eps={train_eps} "))
        if cfg.attack.train.norm == "linf":
            train_gamma = 1.0 * cfg.attack.train.gamma / cfg.dataset.max_color_value
            assert train_gamma <= 1.0
            adversary = LinfPGDAttack(
                predict=model,
                loss_fn=criterion,
                eps=eps_schedule(epoch) if eps_schedule else train_eps,
                nb_iter=cfg.attack.train.step,
                eps_iter=train_gamma,
                rand_init=cfg.attack.train.random_init,
                clip_min=0.0,
                clip_max=1.0,
                targeted=False,
            )
        else:
            raise NotImplementedError

    for i, (input, target) in enumerate(train_loader):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        N, C, W, H = input.shape
        assert C == 3
        total_batch = len(train_loader)

        ## inner maximization (move normalization into the model)
        if cfg.train_test.mode.lower() == "fat":
            batch_shape = (N, C, W, H)
            if cfg.attack.train.random_init:
                init_noise = sampler.sample(batch_shape).to(device)
            else:
                init_noise = torch.zeros(batch_shape).to(device)

            # update lr only for FAT
            lr = lr_schedule(epoch + (i + 1) / total_batch)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # fgsm for FAT
            batch_noise = init_noise.clone().detach().requires_grad_(True)
            input_adv = input + batch_noise
            input_adv.clamp_(0.0, 1.0)
            output = model(input_adv)

            loss = criterion(output, target)
            if cfg.train_test.half:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # 1-iter pdg attack
            fgsm_noise = torch.sign(batch_noise.grad) * train_eps
            init_noise += fgsm_noise.data
            init_noise.clamp_(-test_eps, test_eps)

            batch_noise = init_noise.clone().detach().requires_grad_(False)
            input_adv = input + batch_noise
            input.clamp_(0.0, 1.0)
        else:
            input_adv = adversary.perturb(input, target)
        lr = optimizer.param_groups[0]["lr"]

        ## outer minimization
        output = model(input_adv)
        loss = criterion(output, target)

        # optimizer
        optimizer.zero_grad()
        if cfg.train_test.half:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        ## record results and elapsed time
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1, N)
        top5.update(prec5, N)
        batch_time.update(time.time() - mark_time)

        mark_time = time.time()

        # log results
        if i % cfg.train_test.print_freq == 0:
            log.info(
                f"Train Epoch: [{epoch}][{i}/{total_batch}]  Time {batch_time.val:.3f} ({batch_time.avg:.3f})  Loss {losses.val:.4f} ({losses.avg:.4f})  Prec@1 {top1.val:.3f} ({top1.avg:.3f})  Prec@5 {top5.val:.3f} ({top5.avg:.3f})  LR {lr:.3f}"
            )

    # update lr schedule for non-FAT
    if cfg.train_test.mode.lower() != "fat":
        lr_schedule.step(epoch + 1)

    if cfg.visualization.tool == "wandb" and cfg.visualization.entity:
        wandb.log({"Train top1 accuracy": top1.avg}, step=epoch)
        wandb.log({"Train top5 accuracy": top5.avg}, step=epoch)
        wandb.log({"Train loss": losses.avg}, step=epoch)


def test_natural(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: DictConfig,
    log: logging.Logger,
) -> float:
    # initialize all meters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()  # eval mode
    mark_time = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            N = input.shape[0]
            output = model(input)
            loss = criterion(output, target)

        ## record results and elapsed time
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1, N)
        top5.update(prec5, N)
        batch_time.update(time.time() - mark_time)

        mark_time = time.time()

        # log results
        if i % cfg.train_test.print_freq == 0:
            log.info(
                f"Test: [{i}/{len(val_loader)}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})  Loss {losses.val:.4f} ({losses.avg:.4f})  Prec@1 {top1.val:.3f} ({top1.avg:.3f})  Prec@5 {top5.val:.3f} ({top5.avg:.3f})"
            )

    log.info(f"Final Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}")
    return top1.avg


def test_pgd(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: DictConfig,
    log: logging.Logger,
):
    ## attack
    if cfg.attack.test.norm == "linf":
        adversary = LinfPGDAttack(
            model,
            loss_fn=criterion,
            eps=1.0 * cfg.attack.test.eps / cfg.dataset.max_color_value,
            nb_iter=cfg.attack.test.step,
            eps_iter=1.0 * cfg.attack.test.gamma / cfg.dataset.max_color_value,
            rand_init=cfg.attack.test.random_init,
            clip_min=0.0,
            clip_max=1.0,
            targeted=False,
        )
    else:
        raise NotImplementedError

    ## initialize all meters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    log.info(
        pad_str(
            f" PGD eps: {cfg.attack.test.eps}, step: {cfg.attack.test.step}, gamma: {cfg.attack.test.gamma}, restarts: {cfg.attack.test.restart} "
        )
    )

    model.eval()  # eval mode
    mark_time = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        ## adversarial inputs
        for j in range(cfg.attack.test.restart):
            input_adv = adversary.perturb(input, target)
            with torch.no_grad():
                if j == 0:
                    final_input_adv = input_adv
                else:
                    # record misclassified images
                    I = output.max(1)[1] != target
                    final_input_adv[I] = input[I]

        with torch.no_grad():
            N = input.shape[0]
            output = model(input_adv)
            loss = criterion(output, target)

        ## record results and elapsed time
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1, N)
        top5.update(prec5, N)
        batch_time.update(time.time() - mark_time)
        mark_time = time.time()

        # log results
        if i % cfg.train_test.print_freq == 0:
            log.info(
                f"PGD Test: [{i}/{len(val_loader)}]  Time {batch_time.val:.3f} ({batch_time.avg:.3f})  Loss {losses.val:.4f} ({losses.avg:.4f})  Prec@1 {top1.val:.3f} ({top1.avg:.3f})  Prec@5 {top5.val:.3f} ({top5.avg:.3f})"
            )

    log.info(f" PGD Final Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}")
