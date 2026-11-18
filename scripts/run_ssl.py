import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.config import SEED,SSL_BATCH_SIZE,SSL_CKPT,SSL_EPOCHS,SSL_LR,SSL_WD,TAU
from src.utils import set_seed,get_device,load_checkpoint
from src.data import DT,get_file_lists,make_loader
from src.models import Encoder
from src.train_ssl import train_ssl

set_seed(SEED)
device=get_device()

ssl_list,_,_,_=get_file_lists()

dataset=DT(ssl_list,label_present=False)
loader=make_loader(dataset,SSL_BATCH_SIZE)

model=Encoder().to(device)

if os.path.exists(SSL_CKPT):
    load_checkpoint(model,SSL_CKPT,device)
    print("loaded weights from path: ",SSL_CKPT)

optim=torch.optim.AdamW(model.parameters(),lr=SSL_LR,weight_decay=SSL_WD)
train_ssl(model,loader,optim,SSL_EPOCHS,device,TAU,SSL_CKPT)




