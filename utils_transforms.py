from torchvision import transforms


# color distortion composed by color jittering and color dropping.
# See Section A of SimCLR: https://arxiv.org/abs/2002.05709
def get_color_distortion(s=1.0):
    # s is the strength of color distortion
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


def get_cifar10_transforms(s=0.5):
    # use strength 0.5 for cifar10. See Section B.9 of SimCLR: https://arxiv.org/abs/2002.05709
    # No data-specific normalization used, and keep inputs in [0, 1].
    color_distort = get_color_distortion(s)
    train_transform = transforms.Compose([transforms.RandomSizedCrop(32),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          color_distort,
                                          transforms.ToTensor()])
    test_transform = transforms.ToTensor()
    return train_transform, test_transform
