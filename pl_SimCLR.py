import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, resnet34, resnet50
import pytorch_lightning as pl
from PIL import Image


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


class SimCLR(pl.LightningModule):
    def __init__(self, backbone='resnet18', batch_size=256):
        super().__init__()
        self.batch_size = batch_size
        model = eval(backbone)(pretrained=False)

        # use 3x3 rather than 7x7, because size of cifar10 is much smaller than ImageNet
        model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        model.maxpool = nn.Identity()

        mlp_dim = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(mlp_dim, mlp_dim),
                                 nn.ReLU(),
                                 nn.Linear(mlp_dim, mlp_dim))

        self.model = model.cuda()

    def forward(self, x):
        return self.model(x)

    def training_step(self, x):
        sizes = x.size()
        x = x.view(sizes[0] * 2, sizes[2], sizes[3], sizes[4]).cuda()

        loss = nt_xent(self(x))
        tensorboard_logs = {'SimCLR_loss': loss}
        return {'SimCLR_loss': loss, 'log': tensorboard_logs}

    def validation_step(self):
        pass

    def test_step(self):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        import torchvision.transforms as transforms
        data_dir = 'data'  # hydra.utils.to_absolute_path(args.data_dir)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        train_set = CustomCIFAR10(root=data_dir, train=True, transform=transform, download=True)

        return DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=16)


if __name__ == '__main__':
    model = SimCLR(backbone='resnet18', batch_size=256)
    trainer = pl.trainer(gpus=2, max_epochs=300)
    trainer.fit()
