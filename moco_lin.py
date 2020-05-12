#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import hydra
from omegaconf import DictConfig
import logging

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18, resnet34
import torch.nn.functional as F
from utils import AverageMeter
from tqdm import tqdm


logger = logging.getLogger(__name__)


@hydra.main(config_path='moco_config.yml')
def main(args: DictConfig) -> None:
    # Prepare model
    logger.info("=> creating model '{}'".format(args.backbone))
    model = eval(args.backbone)(pretrained=False)

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
    state_dict = torch.load('moco_{}_epoch{}.pt'.format(args.backbone, args.load_epoch), map_location="cpu")
    # rename moco pre-trained keys
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
            # remove prefix
            state_dict[k[len("encoder_q"):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

    logger.info("=> loaded pre-trained model moco_{}_epoch{}.pt".format(args.backbone, args.load_epoch))
    model = model.cuda()

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    optimizer = torch.optim.Adam(parameters, lr=0.001, weight_decay=0.)

    cudnn.benchmark = True

    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_ = transforms.Compose([
        transforms.ToTensor(),
    ])

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=False)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=transform_, download=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,  drop_last=False)

    optimal_loss, optimal_acc = 1e5, 0.
    for epoch in range(1, args.finetune_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, epoch, optimizer)
        test_loss, test_acc = run_epoch(model, test_loader, epoch)

        if train_loss < optimal_loss:
            optimal_loss = train_loss
            optimal_acc = test_acc
            logger.info("==> New best results")
            torch.save(model.state_dict(), "moco_lin_{}_best.pt".format(args.backbone))

    logger.info("Best test acc: {:.4f}".format(optimal_acc))


def run_epoch(model, dataloader, epoch, optimizer=None):
    loss_meter = AverageMeter('Loss')
    acc_meter = AverageMeter('Acc')

    if optimizer:
        model.train()
    else:
        model.eval()

    loader_bar = tqdm(dataloader)
    for x, y in loader_bar:
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

        if optimizer:
            loader_bar.set_description("Train epoch {}, loss: {:.4f}, acc: {:.4f}"
                                       .format(epoch, loss_meter.avg, acc_meter.avg))
        else:
            loader_bar.set_description("Test epoch {}, loss: {:.4f}, acc: {:.4f}"
                                       .format(epoch, loss_meter.avg, acc_meter.avg))

    return loss_meter.avg, acc_meter.avg


if __name__ == '__main__':
    main()
