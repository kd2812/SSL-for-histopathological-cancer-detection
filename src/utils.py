import os
import numpy as np
import torch
import random


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(42)
    torch.cuda.manual_seed(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model,path):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    torch.save(model.state_dict(),path)

def load_checkpoint(model,path,device):
    model.load_state_dict(torch.load(path,weights_only=True))
    return model.to(device)

def num(v):
    """Turn a CSV cell like "tensor(0.84, device='cuda:0')" or 0.84 into a float."""
    return float(str(v).replace("tensor(", "").split(",")[0].replace(")", ""))
