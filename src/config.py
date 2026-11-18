"""This is the file containing all the seeds, hyperparameters and paths."""
import os

# -------------- Paths ---------------


SAVE_DIR=os.environ.get("SSL_SAVE_DIR","/content/drive/MyDrive/Colab Notebooks/SSL_histo_cancer_detect")

DATA_DIR=os.environ.get("SSL_DATA_DIR","/root/.cache/kagglehub/competitions/histopathologic-cancer-detection")

SSL_CKPT=os.path.join(SAVE_DIR,"SSLEncoder.pth")
SSL_METRICS_CSV = os.path.join(SAVE_DIR, "Model_metrics.csv")
BASELINE_METRICS_CSV = os.path.join(SAVE_DIR, "BL_metrics.csv")


def head_ckpt(percent):
    return os.path.join(SAVE_DIR, f"Final_Layer_p{percent}.pth")


BASELINE_ENCODER_CKPT=os.path.join(SAVE_DIR,"BLEncoder.pth")
BASELINE_FINAL_CKPT=os.path.join(SAVE_DIR,"BLFinal_Layer.pth")



# -------------Reproducibility---------------
SEED=42


# ------------- DATA SPLIT ------------------

N_SSL_FROM_TRAIN=80000
N_PROBE_END=150000

# ------------- LOADERS ---------------------

SSL_BATCH_SIZE=64
HEAD_BATCH_SIZE=64
EVAL_BATCH_SIZE=128
NUM_WORKERS=2

# ------------- SSL PRE-TRAINING ----------------
TAU=0.5
SSL_LR=5e-5
SSL_WD=1e-5
SSL_EPOCHS=2

# -------------- HEAD / BASELINE PRETRAINING -------------
HEAD_LR=1e-5
HEAD_WD=1e-4
HEAD_EPOCHS=5
LABEL_PERCENT=[1,5,10,100]
