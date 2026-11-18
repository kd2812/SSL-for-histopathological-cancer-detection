import matplotlib.pyplot as plt
import numpy as numpy
import pandas as pd
import os
from src.config import SSL_METRICS_CSV,BASELINE_METRICS_CSV,SAVE_DIR

def efficiency_plots(ssl_csv=SSL_METRICS_CSV,bl_csv=BASELINE_METRICS_CSV,save_dir=SAVE_DIR):
        
    DIR =save_dir
    def num(v):  # "tensor(0.84, device='cuda:0')" -> 0.84
        return float(str(v).replace("tensor(", "").split(",")[0].replace(")", ""))

    bl = pd.read_csv(os.path.join(DIR,"BL_metrics.csv"), index_col=0)
    ssl = pd.read_csv(os.path.join(DIR,"Model_metrics.csv"), index_col=0)
    x = [int(c) for c in ssl.columns]

    for m in ["AUC-ROC", "F1 Score", "Recall", "Specificity", "ECE", "Accuracy"]:
        plt.figure(figsize=(5, 4))
        plt.plot(x, [num(v) for v in ssl.loc[m]], "o-", label="SSL")
        plt.axhline(num(bl[m].iloc[0]), color="k", ls="--", label="Supervised 100%")
        plt.xscale("log")
        plt.xticks(x, x)
        plt.xlabel("Labels used (%)")
        plt.ylabel(m)
        plt.legend()
        plt.savefig(os.path.join(DIR,f"efficiency_{m}.png"), dpi=200)
        plt.show()