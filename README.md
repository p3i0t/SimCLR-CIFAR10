# Self-Supervised Learning on CIFAR-10 
This repo contains implementations of [SimCLR](https://arxiv.org/abs/2002.05709), [MoCo version 2](https://arxiv.org/abs/2003.04297) (the improved
version of MoCo) and experimental results on CIFAR10.  This repo aims to facilitate the fast proof-of-concept and research based on
these two self-supervised learning frameworks. Thus I try to keep it clean and minimal, and avoid over-engineering. 
All experiments could be run on only 1 single GPU (1080Ti). After 1000 training epochs, we could get 9x.xx% test accuracy
 with SimCLR, and 9x.xx% with MoCo v2.

## Dependencies
pytorch 1.5


## Usage

### [SimCLR](https://arxiv.org/abs/2002.05709)
Train SimCLR with  ``resnet18`` as backbone:

``python simclr.py backbone=resnet18``

Linear evaluation:

``python simclr_lin.py backbone=resnet18``
The default ``batch_size`` is 512. All the hyperparameters are available in ``simclr_config.yml``,
 which could be overrided from the command line.

### [MoCo v2](https://arxiv.org/abs/2003.04297)
Train MoCo v2 with  ``resnet18`` as backbone :

``python moco.py backbone=resnet18 batch_size=256 moco_k=4096 moco_t=0.2 ``

Linear evaluation:

``python simclr_lin.py backbone=resnet18 batch_size=256 ``

The default ``batch_size`` is 256. All the hyperparameters are available in ``moco_config.yml``,
 which could be overrided from the command line.


## Experimental results

### Memory and Epoch Time
We could train SimCLR on one 1080Ti GPU (11G memory) with ``resnet18`` and ``resnet34``(not enough
memory for resnet50).

|Evaluation| Batch Size| Backbone |Projection Dim|Training Epochs| Memory | s/Epoch | Test Acc|
|----|----|----|-----|----|----|----|---|
|Linear Finetune|512|resnet18|128|1000| ~6.2G| 42s||
|Linear Finetune|512|resnet34|128|1000| ~9.7G| 64s||
|Linear Finetune|512|resnet50|128|1000| -| -|-|


