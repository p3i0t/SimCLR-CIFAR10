#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import builtins
import math
import os

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# import moco.loader
from utils import AverageMeter
import moco.builder
from PIL import Image

import hydra
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


@hydra.main(config_path='config_moco.yml')
def main(args: DictConfig) -> None:
    """
    One node (machine), N GPU cards. So world_size=1 is adjusted to world_size * N, the # processes.
    """
    n_gpus = torch.cuda.device_count()
    args.world_size = n_gpus * args.world_size
    # Use torch.multiprocessing.spawn to launch distributed processes: the main_worker process function
    mp.spawn(main_worker, nprocs=n_gpus, args=(args, ))


class CustomCIFAR10(datasets.CIFAR10):
    def __init__(self, **kwds):
        super().__init__(**kwds)

    def __getitem__(self, idx):
        if not self.train:
            return super().__getitem__(idx)

        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        return self.transform(img), self.transform(img)
        # imgs = [self.transform(img), self.transform(img)]
        # return torch.stack(imgs)  # stack a positive pair


def main_worker(rank, args):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '1234'

    torch.cuda.set_device(rank)  # put the model and training on GPU rank.

    dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=rank)

    # create model
    logger.info("=> creating model '{}'".format(args.backbone))
    model = moco.builder.MoCo(
        models.__dict__[args.backbone],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    logger.info(model)
    model = model.cuda()
    model = DDP(model, device_ids=[rank], output_device=rank)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # data_dir = hydra.utils.to_absolute_path(args.data_dir)
    train_dataset = CustomCIFAR10(root='../../../data', train=True, transform=transform, download=True)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=16, pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)
        if epoch >= 100 and epoch % 100 == 0:
            checkpoint = {
                'epoch': epoch,
                'backbone': args.backbone,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }

            torch.save(checkpoint, "checkpoint_{}.pt".format(epoch))


def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss')
    acc = AverageMeter('Acc')

    # switch to train mode
    model.train()
    for batch_id, (x_a, x_b) in enumerate(train_loader):
        if args.gpu is not None:
            x_a = x_a.cuda(args.gpu, non_blocking=True)
            x_b = x_b.cuda(args.gpu, non_blocking=True)

        # compute output
        output, target = model(im_q=x_a, im_k=x_b)
        loss = criterion(output, target)

        # acc are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1 = (output.argmax(dim=1) == target).float().mean().item()

        acc.update(acc1, x_a.size(0))
        losses.update(loss.item(), x_a.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logger.info("Epoch {}, loss: {:.4f}, acc: {:.4f}".format(epoch, losses.avg, acc.avg))


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
