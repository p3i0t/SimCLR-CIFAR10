#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from utils import AverageMeter


import hydra
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


@hydra.main(config_path='config_lin.yml')
def main(args: DictConfig) -> None:
    # create model
    logger.info("=> creating model '{}'".format(args.backbone))
    model = models.__dict__[args.backbone]()

    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # load pre-trained
    checkpoint = torch.load('checkpoint_{}.pt'.format(args.load_epoch), map_location="cpu")
    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

    logger.info("=> loaded pre-trained model checkpoint_{}.pt".format(args.load_epoch))
    model = model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias
    # optimizer = torch.optim.SGD(parameters, 0.1,
    #                             momentum=args.momentum,
    #                             weight_decay=0.)
    optimizer = torch.optim.SGD(parameters, lr=30, weight_decay=0., momentum=0.9)

    cudnn.benchmark = True

    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=transform_, download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,  drop_last=False)

    optimal_loss = 1e5
    for epoch in range(1, args.finetune_epochs + 1):
        adjust_learning_rate(optimizer, epoch, args)
        loss, acc = run_epoch(model, train_loader, optimizer)
        logger.info("Epoch {}, train loss: {:.4f}, acc: {:.4f}".format(epoch, loss, acc))
        if loss < optimal_loss:
            optimal_loss = loss
            logger.info("==> New best results")
            checkpoint = {
                'epoch': epoch,
                'backbone': args.backbone,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, "lin_checkpoint_best.pt")

        loss, acc = run_epoch(model, test_loader)
        logger.info("test loss: {:.4f}, acc: {:.4f}".format(loss, acc))


def run_epoch(model, dataloader, optimizer=None):
    loss_meter = AverageMeter('Loss')
    acc_meter = AverageMeter('Acc')

    if optimizer:
        model.train()
    else:
        model.eval()
    for batch_id, (x, y) in enumerate(dataloader):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        acc = (logits.argmax(dim=1) == y).float().mean().item()
        acc_meter.update(acc, x.size(0))
        loss_meter.update(loss.item(), x.size(0))
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return loss_meter.avg, acc_meter.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
