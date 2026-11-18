import torch
import torchvision
import torchvision.transforms.v2 as T


# -------- data augmentations used --------------
CJ=T.RandomApply([T.ColorJitter(0.4,0.4,0.2,0.1)],p=0.8)
HINV=T.RandomHorizontalFlip(p=0.5)
VINV=T.RandomVerticalFlip(p=0.5)
BL=T.RandomApply([T.GaussianBlur(9,(0.1,1.0))],p=0.5)
norm=T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])

# --------- denormalizing function -----------------
def denorm(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)


# --------- transformation pipeline -------------------
transformation_pipeline=torchvision.transforms.Compose([
    CJ,
    HINV,
    VINV,
    BL,
])


# ------------ function for transformation---------------

def data_augmentation(z):
    z=transformation_pipeline(z)

    z=norm(z)
    return z