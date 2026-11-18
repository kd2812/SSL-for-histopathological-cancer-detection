import numpy as np
import pandas as pd
import torchmetrics.functional as TMF
import os
from src.config import SAVE_DIR
import matplotlib.pyplot as plt


def ece(y_real,y_result,bins=10):
    bin_boundaries = np.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    y_res_copy=y_result.clone().detach().cpu().numpy()
    y_res_copy_labels=np.where(y_res_copy>=0.5,1,0)
    y_real_copy=y_real.clone().cpu().numpy()

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
      if bin_upper==1:
        in_bin=(y_res_copy>=bin_lower)
      else:
        in_bin=(y_res_copy>=bin_lower) & (y_res_copy<bin_upper)

      y_res_bin=y_res_copy[in_bin]
      y_real_bin=y_real_copy[in_bin]
      y_res_labels_bin=y_res_copy_labels[in_bin]
      if len(y_res_bin)==0:continue
      acc=np.mean(y_real_bin)
      conf=y_res_bin.mean()
      ece += np.abs(acc - conf)* len(y_res_bin)
    return ece/len(y_real)

def reliability_plot(y_real,y_result,bins=10,save_img=False,title="Reliability-curve",save_dir=SAVE_DIR):
    bin_boundaries = np.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    y_res_copy=y_result.clone().detach().cpu().numpy()
    y_res_copy_labels=np.where(y_res_copy>=0.5,1,0)
    y_real_copy=y_real.clone().cpu().numpy()
    reliability_dict={}
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
      if bin_upper==1:
        in_bin=(y_res_copy>=bin_lower)
      else:
        in_bin=(y_res_copy>=bin_lower) & (y_res_copy<bin_upper)

      y_res_bin=y_res_copy[in_bin]
      y_real_bin=y_real_copy[in_bin]
      y_res_labels_bin=y_res_copy_labels[in_bin]
      if len(y_res_bin)==0:continue
      acc=np.mean(y_real_bin)
      conf=y_res_bin.mean()
      reliability_dict[conf]=acc
    reliability_array=np.array(list(reliability_dict.items()))
    plt.figure(figsize=(5,5))
    plt.plot(reliability_array[:,0],reliability_array[:,1],"o--")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("Average Predicted Probability")
    plt.ylabel("Fraction of positives")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title(title)
    if save_img:
        os.makedirs(save_dir,exist_ok=True)
        addr=os.path.join(save_dir,f"{title}.png")
        plt.savefig(addr)
    plt.show()
    plt.close()

def metrics_display(y_real,y_result):
    acc=TMF.accuracy(y_result, y_real, task='binary')
    f1=TMF.f1_score(y_result, y_real, task='binary')
    roc=TMF.auroc(y_result, y_real,task='binary')
    spec=TMF.specificity(y_result, y_real,task='binary')
    recall=TMF.recall(y_result, y_real,task='binary')
    ece_score=ece(y_real,y_result)
    print(f"Accuracy : {acc}")
    print(f"F1 Score : {f1}")
    print(f"AUC-ROC : {roc}")
    print(f"Specificity : {spec}")
    print(f"Recall : {recall}")
    print(f"ECE : {ece_score}")

def metrics_dict_creator(y_real,y_result):
    acc=TMF.accuracy(y_result, y_real, task='binary')
    f1=TMF.f1_score(y_result, y_real, task='binary')
    roc=TMF.auroc(y_result, y_real,task='binary')
    spec=TMF.specificity(y_result, y_real,task='binary')
    recall=TMF.recall(y_result, y_real,task='binary')
    ece_score=ece(y_real,y_result)
    return {
        "Accuracy" : acc,
        "F1 Score" : f1,
        "AUC-ROC" : roc,
        "Specificity" : spec,
        "Recall" : recall,
        "ECE" : ece_score
    }

