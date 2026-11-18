import kagglehub
import pandas as pd
import numpy as np
import random
from PIL import Image
import os
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms.functional as TF
from src.data_augmentations import norm,data_augmentation
from config import DATA_DIR,N_PROBE_END,N_SSL_FROM_TRAIN,NUM_WORKERS


# --------- Downloading data-----------------
def download_data():
    """Needs Kaggle credentials in the environment"""
    import kagglehub

    return kagglehub.competition_download("histopathologic-cancer-detection")

# ---------- Getting File lists for training and testing --------------
def get_file_lists(data_dir=DATA_DIR):
    train_list=[]
    layer_list=[]
    test_list=[]
    labels={}
    train_labels_dir=os.path.join(DATA_DIR,"train_labels.csv")
    labels_df=pd.read_csv(train_labels_dir)
    labels=pd.Series(labels_df.label.values,index=labels_df.id)
    test_dir=os.path.join(DATA_DIR,"test")
    for file in sorted(os.listdir(test_dir)):
                if file.endswith(".tif"):
                    train_list.append(f"{test_dir}/{file}")
    i=0
    train_dir=os.path.join(DATA_DIR,"train")
    for file in sorted(os.listdir(train_dir)):
                if file.endswith(".tif"):
                    i+=1
                    if(i<=N_SSL_FROM_TRAIN):
                        train_list.append(f"{train_dir}/{file}")
                    elif(i<N_PROBE_END):
                        layer_list.append(f"{train_dir}/{file}")
                    else:
                        test_list.append(f"{train_dir}/{file}")

    print("Train List Size: ",len(train_list))
    print("Layer List Size: ",len(layer_list))
    print("Test List Size: ",len(test_list))

    return train_list,layer_list,test_list,labels

# ---------- Dataset Class ---------------------------

class DT(Dataset):
    def __init__(self,train,label_present,labels=None,p=100):
        if label_present and p<100:
            self.indexes=random.sample(range(0,len(train)),(len(train)//100)*p)
        # self.input_dir=input_dir
        if label_present:
            if p==100:
                self.img_address=train
            else:
                self.img_address=[train[i] for i in self.indexes]
        else:
            self.img_address=train

        self.labels_present=label_present
        self.labels=labels
        print(f"{len(self.img_address)} images loaded successfully")
    def __len__(self):
        return len(self.img_address)
    def __getitem__(self,idx):
        img_add=self.img_address[idx]
        img=Image.open(img_add).convert("RGB")
        # img=Image.open(img_add)
        if self.labels_present:
            y=self.labels[img_add[-44:-4:1]]
            return norm(TF.to_tensor(img)),y

        return norm(TF.to_tensor(img)),data_augmentation(TF.to_tensor(img))


# ------------------ data loader function ----------------------------

def make_loader(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        persistent_workers=NUM_WORKERS > 0,
    )
