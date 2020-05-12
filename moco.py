#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import Image

import hydra
from omegaconf import DictConfig
import logging

import numpy as np

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, resnet34

from models import MoCo
from utils import AverageMeter
from utils_transforms import get_cifar10_transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CustomCIFAR10(CIFAR10):
    def __getitem__(self, idx):  # generate mini-batch pairs.
        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        return self.transform(img), self.transform(img)


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


@hydra.main(config_path='moco_config.yml')
def main(args: DictConfig) -> None:
    assert torch.cuda.is_available()
    cudnn.benchmark = True

    train_transform, test_transform = get_cifar10_transforms(s=0.5)
    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    train_dataset = CustomCIFAR10(root=data_dir, train=True, transform=train_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=16, pin_memory=True, drop_last=True)

    # Prepare model
    assert args.backbone in ['resnet18', 'resnet34']
    model = MoCo(eval(args.backbone), args.moco_dim, args.moco_k, args.moco_m, args.moco_t).cuda()
    logger.info("Base encoder: {}".format(args.backbone))

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # cosine annealing lr
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            args.epochs * len(train_loader),
            args.learning_rate,  # lr_lambda computes multiplicative factor
            1e-4))

    for epoch in range(1, args.epochs + 1):
        # train for one epoch
        loss_meter = AverageMeter('Loss')
        acc_meter = AverageMeter('Acc')

        # switch to train mode
        model.train()
        loader_bar = tqdm(train_loader)
        for x_a, x_b in loader_bar:
            x_a = x_a.cuda(non_blocking=True)
            x_b = x_b.cuda(non_blocking=True)

            # compute output
            logits, labels = model(im_q=x_a, im_k=x_b)
            loss = F.cross_entropy(logits, labels)

            # acc are (K+1)-way contrast classifier accuracy
            # measure accuracy and record loss
            acc = (logits.argmax(dim=1) == labels).float().mean().item()

            acc_meter.update(acc, x_a.size(0))
            loss_meter.update(loss.item(), x_a.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loader_bar.set_description(
                "Train epoch {}, Loss: {:.4f}, Acc: {:.4f}".format(epoch, loss_meter.avg, acc_meter.avg))

        if epoch >= args.log_interval and epoch % args.log_interval == 0:
            logger.info("==> Save checkpoint. Train epoch {}, Loss: {:.4f}, Acc: {:.4f}"
                        .format(epoch, loss_meter.avg, acc_meter.avg))

            torch.save(model.state_dict(), "moco_{}_epoch{}.pt".format(args.backbone, epoch))


if __name__ == '__main__':
    main()
