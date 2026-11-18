import torch
from src.loss_fnc import NTXent_Loss
from src.utils import save_checkpoint
import matplotlib.pyplot as plt
import os
def train_ssl(encoder,loader,optimizer,epochs,device,tau=0.5,ckpt_path=None):
    """returns the history of training the SSL Encoder and """
    loss_dict={}
    min_loss=None
    for epoch in range(epochs):
        c=0
        l=[]
        for v1,v2 in loader:
            v1,v2=v1.to(device),v2.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', enabled=(device.type == "cuda")):
                h1,z1=encoder(v1)
                h2,z2=encoder(v2)

                z=torch.stack([z1,z2],dim=1).reshape(-1,z1.shape[-1])

                loss=NTXent_Loss(z,tau=0.5)

                loss.backward()

                optimizer.step()

            l.append(loss.item())

            c+=1
            print("|",end=" ")
            if c%10==0:
                print("\n")
                print(f"Epoch : {epoch} # {c}")
                print(f"Loss: {loss.item()}")
                print("-"*20)
                if(min_loss is None or loss.item()<min_loss):
                    min_loss=loss.item()
                    ckpt_path_encoder=os.path.join(ckpt_path,"SSLEncoder")
                    save_checkpoint(encoder,ckpt_path_encoder)
                    # torch.save(encoder.state_dict(),"/content/drive/MyDrive/Colab Notebooks/SSL_histo_cancer_detect/SSLEncoder.pth")
                    print(f"saved SSLEncoder.pth, path : {ckpt_path_encoder}")
                    print("-"*20)
                if c%100==0:
                    plt.plot(l)
                    plt.show()
        loss_dict[epoch]=l
    return loss_dict