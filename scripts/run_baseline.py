import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SAVE_DIR,SEED,HEAD_BATCH_SIZE,HEAD_EPOCHS,HEAD_LR,HEAD_WD,BASELINE_ENCODER_CKPT,BASELINE_FINAL_CKPT
from src.data import DT,get_file_lists,make_loader
from src.models import Encoder,Linear_Layer
from src.train_head import train_head
from src.utils import get_device,save_checkpoint,set_seed
from src.metrics import metrics_dict_creator,metrics_display,reliability_plot
set_seed(SEED)

device=get_device()
_,linear_list,_,labels=get_file_lists()

loader=make_loader(DT(linear_list,True,labels,p=100),HEAD_BATCH_SIZE,True)


print("======== Baseline ==========")
encoder=Encoder().to(device)
head=Linear_Layer().to(device)

history,y_real,y_result=train_head(encoder,head,loader,device,100,HEAD_EPOCHS,HEAD_LR,HEAD_WD,False,SAVE_DIR)

metrics_display(y_real,y_result)
metrics_dict=metrics_dict_creator(y_real,y_result)
metrics_df=pd.DataFrame(metrics_dict)
metrics_df.to_csv(os.path.join(SAVE_DIR,"Baseline_metrics.csv"))
reliability_plot(y_real,y_result,save_img=True,title="Reliability_plot_baseline")

