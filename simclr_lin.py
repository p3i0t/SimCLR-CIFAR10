import hydra
from omegaconf import DictConfig
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

from models import Model
from utils import AverageMeter
from utils_transforms import get_cifar10_transforms

from tqdm import tqdm


logger = logging.getLogger(__name__)


class LinModel(nn.Module):
    """Linear wrapper of encoder."""
    def __init__(self, encoder: nn.Module, feature_dim: int, n_classes: int):
        super().__init__()
        self.enc = encoder
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.lin = nn.Linear(self.feature_dim, self.n_classes)

    def forward(self, x):
        return self.lin(self.enc(x))


def run_epoch(model, dataloader, epoch, optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('acc')
    loader_bar = tqdm(dataloader)
    for x, y in loader_bar:
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = (logits.argmax(dim=1) == y).float().mean()
        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc.item(), x.size(0))

        loader_bar.set_description("Epoch {}, loss: {:.4f}, Acc: {:.4f}".format(epoch, loss_meter.avg, acc_meter.avg))
    return loss_meter.avg, acc_meter.avg


@hydra.main(config_path='simclr_config.yml')
def finetune(args: DictConfig) -> None:
    train_transform = transforms.Compose([transforms.RandomResizedCrop(32),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor()])
    _, test_transform = get_cifar10_transforms(s=0.5)

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    train_set = CIFAR10(root=data_dir, train=True, transform=train_transform, download=False)
    test_set = CIFAR10(root=data_dir, train=False, transform=test_transform, download=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Prepare model
    pre_model = Model(projection_dim=args.projection_dim).cuda()
    pre_model.load_state_dict(torch.load('simclr_epoch{}.pt'.format(args.load_epoch)))
    model = LinModel(pre_model.enc, feature_dim=pre_model.feature_dim, n_classes=len(train_set.targets))

    # Fix encoder
    model.enc.requires_grad = False
    parameters = [param for param in model.parameters() if param.requires_grad is True]  # trainable parameters.
    optimizer = torch.optim.SGD(
        parameters,
        lr=0.2,  # use larger lr=0.1 * batch_size / 256. See Section B.7 of SimCLR paper.
        momentum=args.momentum,
        weight_decay=0.0,   # no decay
        nesterov=True)

    # scheduler = LambdaLR(
    #     optimizer,
    #     lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
    #         step,
    #         args.epochs * len(train_loader),
    #         0.1,  # lr_lambda computes multiplicative factor
    #         1e-6))

    optimal_loss, optimal_acc = 1e5, 0.
    for epoch in range(1, args.finetune_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, epoch, optimizer)
        test_loss, test_acc = run_epoch(model, test_loader, epoch)

        if train_loss < optimal_loss:
            optimal_loss = train_loss
            optimal_acc = test_acc

    logger.info("Best Test Acc: {:.4f}".format(optimal_acc))


if __name__ == '__main__':
    finetune()


