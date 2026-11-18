import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.config import SEED,BASELINE_METRICS_CSV, EVAL_BATCH_SIZE, LABEL_PERCENT, SSL_CKPT,SSL_METRICS_CSV,BASELINE_ENCODER_CKPT,BASELINE_FINAL_CKPT,head_ckpt
from src.data import DT, get_file_lists, make_loader
from src.models import Encoder, Linear_Layer
from src.train_head import train_head
from src.utils import get_device, load_checkpoint, save_checkpoint, set_seed
from src.metrics import metrics_dict_creator,metrics_display,reliability_plot
from src.eval import evaluate
set_seed(SEED)
device=get_device()


_,_,eval_list,labels=get_file_lists()

loader=make_loader(DT(eval_list,True,labels,p=100),EVAL_BATCH_SIZE,False)

# ---- SSL ----
ssl_encoder = load_checkpoint(Encoder(), SSL_CKPT, device)
ssl_results = {}
for p in LABEL_PERCENT:
    head = load_checkpoint(Linear_Layer(), head_ckpt(p), device)
    ssl_results[p], y, prob = evaluate(ssl_encoder, head, loader, device)
    print(f"SSL {p}%:", ssl_results[p])
    reliability_plot(y, prob, save_img=True, title=f"Reliability-curve_p{p}_eval_data")
pd.DataFrame(ssl_results).to_csv(SSL_METRICS_CSV)

# --------- baseline -----------

bl_results={}
p=100
enc = load_checkpoint(Encoder(), BASELINE_ENCODER_CKPT, device)
head = load_checkpoint(Linear_Layer(), BASELINE_FINAL_CKPT, device)
bl_results[p], y, prob = evaluate(enc, head, loader, device)
print(f"Baseline {p}%:", bl_results[p])
reliability_plot(y, prob, save_img=True, title=f"Baseline_Reliability-curve_p{p}_eval_data")
pd.DataFrame(bl_results).to_csv(BASELINE_METRICS_CSV)


