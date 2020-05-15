# SimCLR on CIFAR-10 
This repo contains implementations of [SimCLR](https://arxiv.org/abs/2002.05709) and experimental results on CIFAR10.
  This repo aims to facilitate the fast proof-of-concept and research based on
SimCLR. So I try to keep it clean and minimal, and avoid over-engineering. All experiments could be run on only 1 single GPU (1080Ti).
 
We get 92.85% test acc with backbone resnet34 on CIFAR10, while the SimCLR paper reports ~93.5% with backbone resnet50.

## Dependencies
pytorch 1.5

hydra

tqdm




## Usage

Train SimCLR with  ``resnet18`` as backbone:

``python simclr.py backbone=resnet18``

Linear evaluation:

``python simclr_lin.py backbone=resnet18``
The default ``batch_size`` is 512. All the hyperparameters are available in ``simclr_config.yml``,
 which could be overrided from the command line.

## Experimental results

We could train SimCLR on one 1080Ti GPU (11G memory) with ``resnet18`` and ``resnet34``(not enough
memory for resnet50).

|Evaluation| Batch Size| Backbone |Projection Dim|Training Epochs| Memory | Training Time /Epoch | Test Acc|Test Acc in Paper|
|----|----|----|-----|----|----|----|---|----|
|Linear Finetune|512|resnet18|128|1000| ~6.2G| 38s|92.06%|[~91%](https://github.com/google-research/simclr)|
|Linear Finetune|512|resnet34|128|1000| ~9.7G| 64s|92.85%|-|
|Linear Finetune|512|resnet50|128|1000| -| -|-|~93.5%|

|Optimization|Initial LR|Optimizer|LR Adjustment|Weight Decay|Momentum|Temperature|
|----|----|----|----|----|----|----|
|SimCLR Training|0.6 (0.3 * batch_size / 256)|SGD|Cosine Annealing (to min lr = 0.001)|1e-6|0.9|0.5|
|Linear Finetune|0.2 (0.1 * batch_size / 256)|SGD|Cosine Annealing (to min lr = 0.001)|0.|0.9|-|

