import torch
from torch.optim import Adam
# from lars import LARS
# from simclr import SimCLR


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def load_model(args):
#     model = SimCLR(backbone=args.backbone,
#                    projection_dim=args.projection_dim,
#                    pretrained=args.pretrained,
#                    normalize=args.normalize)
#
#     if args.inference:
#         model.load_state_dict(torch.load("SimCLR_{}_epoch90.pth".format(args.backbone)))
#
#     model = model.to(args.device)
#
#     scheduler = None
#     if args.optimizer == "Adam":
#         optimizer = Adam(model.parameters(), lr=3e-4)  # TODO: LARS
#     elif args.optimizer == "LARS":
#         # optimized using LARS with linear learning rate scaling
#         # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
#         learning_rate = 0.3 * args.batch_size / 256
#         optimizer = LARS(
#             model.parameters(),
#             lr=learning_rate,
#             weight_decay=args.weight_decay,
#             exclude_from_weight_decay=["batch_normalization", "bias"],
#         )
#
#         # "decay the learning rate with the cosine decay schedule without restarts"
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, args.epochs, eta_min=0, last_epoch=-1
#         )
#     else:
#         raise NotImplementedError
#
#     return model, optimizer, scheduler


def save_model(args, model, epoch):
    model_path = "SimCLR_{}_epoch{}.pth".format(args.backbone, epoch)

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)