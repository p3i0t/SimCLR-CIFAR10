import hydra
from omegaconf import DictConfig
import logging
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as tfs
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torchlars import LARS
from utils import AverageMeter
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


class CustomCIFAR10(CIFAR10):
    def __init__(self, **kwds):
        super().__init__(**kwds)

    def __getitem__(self, idx):
        if not self.train:
            return super().__getitem__(idx)

        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        return torch.stack(imgs)  # stack a positive pair


def pair_cosine_similarity(x, eps=1e-8):
    n = x.norm(p=2, dim=1, keepdim=True)
    return (x @ x.t()) / (n * n.t()).clamp(min=eps)


def nt_xent(x, t=0.5):
    x = pair_cosine_similarity(x)
    x = torch.exp(x / t)
    idx = torch.arange(x.size()[0])
    # Put positive pairs on the diagonal
    idx[::2] += 1
    idx[1::2] -= 1
    x = x[idx]
    # subtract the similarity of 1 from the numerator
    x = x.diag() / (x.sum(0) - torch.exp(torch.tensor(1 / t)))
    return -torch.log(x.mean())


def finetune_linear(model, args):
    for param in model.parameters():
        param.requires_grad = False

    normal_transform = tfs.Compose([
        tfs.Resize(32),
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
    ])

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    train_set = CIFAR10(root=data_dir, train=True, transform=normal_transform, download=True)
    test_set = CIFAR10(root=data_dir, train=False, transform=normal_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    mlp_dim = model.fc[0].in_features
    model.fc = nn.Linear(mlp_dim, len(train_set.classes))
    model = model.cuda()

    #  finetune a linear classifier
    optimizer = Adam(model.parameters(), lr=0.1)
    base_optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,  # 0.05 * batch_size / 256
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True)

    optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            args.epochs * len(train_loader),
            0.1,  # lr_lambda computes multiplicative factor
            1e-6))
    criterion = nn.CrossEntropyLoss()

    model.train()
    classification_loss_meter = AverageMeter("classification_loss")
    for epoch in range(1, args.finetune_epochs + 1):
        for batch_id, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            classification_loss_meter.update(loss.item(), x.size(0))
        logger.info("Epoch {}, Linear finetune loss: {:.4f}".format(epoch, classification_loss_meter.avg))

    model.eval()
    acc_meter = AverageMeter("Acc")
    for batch_id, (x, y) in enumerate(test_loader):
        x, y = x.cuda(), y.cuda()
        pred = model(x)
        acc = (pred.argmax(dim=-1) == y).float().mean().item()
        acc_meter.update(acc, x.size(0))

    logger.info("Test Acc: {:.4f}".format(acc_meter.avg))


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


@hydra.main(config_path='config_SimCLR.yml')
def train_SimCLR(args: DictConfig) -> None:
    assert torch.cuda.is_available()

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    transform = tfs.Compose([
        tfs.RandomResizedCrop(32),
        tfs.RandomHorizontalFlip(),
        tfs.ColorJitter(0.5, 0.5, 0.5, 0.5),
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
    ])

    train_set = CustomCIFAR10(root=data_dir, train=True, transform=transform, download=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    model = resnet18(pretrained=False)

    # use 3x3 rather than 7x7, because size of cifar10 is much smaller than ImageNet
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()

    mlp_dim = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(mlp_dim, 2048),
                             nn.ReLU(),
                             nn.Linear(2048, args.projection_dim))

    model = model.cuda()
    if args.load_checkpoint:
        ckpt = torch.load('simclr-e{}.pt'.format(args.load_epoch))
        model.load_state_dict(ckpt['model'])
        logger.info("Model loaded on epoch {}".format(args.load_epoch))
        finetune_linear(model, args)
    else:
        base_optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True)

        optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
                step,
                args.epochs * len(train_loader),
                args.learning_rate,  # lr_lambda computes multiplicative factor
                1e-6))

        # SimCLR training
        model.train()
        for epoch in range(1, args.epochs + 1):
            loss_meter = AverageMeter("SimCLR_loss")
            for x in train_loader:
                sizes = x.size()
                x = x.view(sizes[0] * 2, sizes[2], sizes[3], sizes[4]).cuda()

                optimizer.zero_grad()
                loss = nt_xent(model(x), args.temperature)
                loss.backward()
                optimizer.step()
                scheduler.step()

                loss_meter.update(loss.item(), x.size(0))

            if epoch >= args.log_interval and epoch % args.log_interval == 0:
                logger.info("Epoch {}, SimCLR loss: {:.4f}".format(epoch, loss_meter.avg))
                # Save checkpoint
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, 'simclr-e{}.pt'.format(epoch))


if __name__ == '__main__':
    train_SimCLR()


