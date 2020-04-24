import numpy as np
from tqdm import tqdm_notebook as tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as tfs
from torchvision.datasets import *
from torchvision.models import *


tf_tr = tfs.Compose([
    tfs.RandomResizedCrop(32),
    tfs.RandomHorizontalFlip(),
    tfs.ColorJitter(0.5, 0.5, 0.5, 0.5),
    tfs.ToTensor(),
    tfs.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
])

tf_de = tfs.Compose([
    tfs.Resize(32),
    tfs.ToTensor(),
    tfs.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
])

tf_te = tfs.Compose([
    tfs.Resize(32),
    tfs.ToTensor(),
    tfs.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
])


class CustomCIFAR10(CIFAR10):
    def __init__(self, **kwds):
        super().__init__(**kwds)

    def __getitem__(self, idx):
        if not self.train:
            return super().__getitem__(idx)

        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        return torch.stack(imgs)


ds_tr = CustomCIFAR10(root='data', train=True, transform=tf_tr, download=True)
ds_de = CIFAR10(root='data', train=True, transform=tf_de, download=True)
ds_te = CIFAR10(root='data', train=False, transform=tf_te, download=True)


dl_tr = DataLoader(ds_tr, batch_size=256, shuffle=True)
dl_de = DataLoader(ds_dcudae, batch_size=256, shuffle=True)
dl_te = DataLoader(ds_te, batch_size=256, shuffle=False)

model = resnet50(pretrained=False)
model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
model.maxpool = nn.Identity()


ch = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(ch, ch),
                           nn.ReLU(),
                           nn.Linear(ch, ch))
model.to("cuda")


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


optimizer = Adam(model.parameters(), lr=0.001)


# SimCLR training
model.train()
for epoch in range(100):
    # c, s = 0, 0
    # pBar = tqdm(dl_tr)
    for data in dl_tr:
        d = data.size()
        x = data.view(d[0]*2, d[2], d[3], d[4]).to('cuda')
        optimizer.zero_grad()
        p = model(x)
        loss = nt_xent(p)
        # s = ((s*c)+(float(loss)*len(p)))/(c+len(p))
        # c += len(p)
        # pBar.set_description('Train: '+str(round(float(s),3)))
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), 'cifar10-rn50-mlp-b256-t0.5-e'+str(epoch + 1)+'.pt')

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(ch, len(ds_de.classes))
model.to('cuda')


optimizer = Adam(model.parameters(), lr=0.003)
criterion = nn.CrossEntropyLoss()


model.train()
for i in range(5):
    # c, s = 0, 0
    # pBar = tqdm(dl_de)
    for data in dl_de:
        x, y = data[0].to('cuda'), data[1].to('cuda')
        optimizer.zero_grad()
        p = model(x)
        loss = criterion(p, y)
        # s = ((s*c)+(float(loss)*len(p)))/(c+len(p))
        # c += len(p)
        # pBar.set_description('Train: '+str(round(float(s),3)))
        loss.backward()
        optimizer.step()


model.eval()
model.eval()
acc_list = []
for data in dl_te:
    x, y = data[0].to('cuda'), data[1].to('cuda')
    p = model(x)
    acc = (p.argmax(dim=-1) == y).float().mean().item()
    acc_list.append(acc)

print("Test Acc: {:.4f}".format(np.mean(acc_list)))