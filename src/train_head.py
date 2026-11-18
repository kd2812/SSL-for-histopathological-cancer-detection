import torch 
import torch.nn as nn
import itertools
import matplotlib.pyplot as plt
from src.metrics import metrics_display
import os
from src.utils import save_checkpoint
def train_head(encoder,head,loader,device,lp=100,epochs=5,learning_rate=1e-5,wd=1e-4,freeze_encoder=True,save_dir=None):

    lossFn=nn.BCEWithLogitsLoss()
    encoder.to(device)
    head.to(device)

    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        params = head.parameters()
    else:
        params = itertools.chain(encoder.parameters(), head.parameters())

    optim=torch.optim.adamw(params,lr=learning_rate,weight_decay=wd)
    min_loss=None

    l=[]

    for epoch in range(epochs):
        c=0
        y_result_accumulated=torch.tensor([]).to(device)
        y_real_accumulated=torch.tensor([]).to(device)
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            optim.zero_grad()
            with torch.no_grad():
                h,_=encoder(x)

            y_result=head(h).squeeze(1)
            loss=lossFn(y_result,y.float())
            loss.backward()
            optim.step()
            c+=1
            y_result=torch.sigmoid(y_result)
            l.append(loss.item())
            y_result_accumulated=torch.cat([y_result_accumulated,y_result])
            y_real_accumulated=torch.cat([y_real_accumulated,y])
            if min_loss is None: min_loss=loss.item()
            print("|",end=" ")
            if(c%10==0):
                print("\n")
                print(f"Epoch : {epoch} # {c}")
                print(f"Loss: {loss.item()}")
                # metrics_display(y_real=y,y_result=y_result)
                print("-"*20)
                if(loss.item()<min_loss):
                    min_loss=loss.item()
                    if freeze_encoder:
                        addr=os.path.join(save_dir,f"Final_Layer_p{lp}.pth")
                        save_checkpoint(head,addr)

                    else:
                        addr=os.path.join(save_dir,f"BLFinal_Layer.pth")
                        save_checkpoint(encoder,addr)
                        addr=os.path.join(save_dir,f"BLEncoder.pth")
                        save_checkpoint(head,addr)

                    # torch.save(head.state_dict(),addr)
                if c%100==0:
                    plt.figure(figsize=(5,5))
                    plt.plot(l)
                    plt.show()
                    plt.close()
        print("-"*20)
        print(f"Epoch#{epoch}")
        metrics_display(y_real=y_real_accumulated.long(),y_result=y_result_accumulated)
        print("-"*20)

    return l,y_real_accumulated,y_result_accumulated