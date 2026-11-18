import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""t-SNE, Grad-CAM examples and efficiency curves.
Usage: python scripts/run_analysis.py"""
import os

from src.config import SAVE_DIR, SEED, SSL_CKPT, head_ckpt
from src.data import DT, get_file_lists, make_loader
from src.gradcam import GradCam, show_gradcam
from src.models import Encoder, Linear_Layer
from src.plots import plot_efficiency_curves
from src.tsne import run_tsne
from src.utils import get_device, load_checkpoint, set_seed

set_seed(SEED)
device = get_device()
_, probe_list, eval_list, labels = get_file_lists()
encoder = load_checkpoint(Encoder(), SSL_CKPT, device)

# t-SNE on labeled probe images
tsne_loader = make_loader(DT(probe_list, True, labels, p=100), 64)
run_tsne(encoder, tsne_loader, device, save_path=os.path.join(SAVE_DIR, "tsne.png"))

# Grad-CAM on 5 random eval images, using the 100%-label head
head = load_checkpoint(Linear_Layer(), head_ckpt(100), device)
cam = GradCam(encoder, head, target_layer=encoder.encoder.layer4)
cam_loader = make_loader(DT(eval_list, True, labels, p=5), batch_size=1)
for i, (x, y) in zip(range(5), cam_loader):
    show_gradcam(cam, x, y, alpha=0.6, save_path=os.path.join(SAVE_DIR, f"Grad-CAM_{i}.png"))
# cam.remove()

# Efficiency curves (reads the metrics CSVs written by run_eval.py)
plot_efficiency_curves()