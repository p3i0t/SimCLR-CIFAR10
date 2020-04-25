import hydra
from omegaconf import DictConfig
import logging

import numpy as np
import logging
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as tfs
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, resnet34, resnet50
from utils import AverageMeter

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


@hydra.main(config_path='SimCLR_config.yml')
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

    model = eval(args.backbone)(pretrained=args.pretrained)

    # use 3x3 rather than 7x7, because size of cifar10 is much smaller than ImageNet
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()

    mlp_dim = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(mlp_dim, mlp_dim),
                             nn.ReLU(),
                             nn.Linear(mlp_dim, mlp_dim))

    model = model.cuda()
    # model = nn.DataParallel(model, device_ids=[0, 1]).cuda()

    optimizer = Adam(model.parameters(), lr=0.001)
    if args.load_checkpoint:
        save_path = 'cifar10-rn50-mlp-b256-t0.5-e90.pt'
        # model.load_state_dict(torch.load(save_path))
        model = torch.load(save_path)
        logger.info("model loaded")
    else:
        loss_meter = AverageMeter("SimCLR_loss")

        # SimCLR training
        model.train()
        for epoch in range(args.epochs):
            for x in train_loader:
                sizes = x.size()
                x = x.view(sizes[0] * 2, sizes[2], sizes[3], sizes[4]).cuda()

                optimizer.zero_grad()
                loss = nt_xent(model(x))
                loss.backward()
                optimizer.step()

                loss_meter.update(loss.item(), x.size(0))

            logger.info("Epoch {}, SimCLR loss: {:.4f}".format(epoch, loss_meter.avg))

            if (epoch + 1) % args.log_interval == 0:
                torch.save(model, 'cifar10-rn50-mlp-b256-t0.5-e' + str(epoch + 1) + '.pt')

    for param in model.parameters():
        param.requires_grad = False

    normal_transform = tfs.Compose([
        tfs.Resize(32),
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
    ])

    train_set = CIFAR10(root=data_dir, train=True, transform=normal_transform, download=True)
    test_set = CIFAR10(root=data_dir, train=False, transform=normal_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model.fc = nn.Linear(mlp_dim, len(train_set.classes))
    model = model.cuda()
    # model = nn.DataParallel(model, device_ids=[0, 1]).cuda()

    #  finetune a linear classifier
    optimizer = Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    model.train()
    classification_loss_meter = AverageMeter("classification_loss")
    for epoch in range(5):
        for batch_id, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            classification_loss_meter.update(loss.item(), x.size(0))

    model.eval()
    acc_meter = AverageMeter("Acc")
    for batch_id, (x, y) in enumerate(test_loader):
        x, y = x.cuda(), y.cuda()
        p = model(x)
        acc = (p.argmax(dim=-1) == y).float().mean().item()
        acc_meter.update(acc, x.size(0))

    logger.info("Test Acc: {:.4f}".format(acc_meter.avg))


if __name__ == '__main__':
    train_SimCLR()


