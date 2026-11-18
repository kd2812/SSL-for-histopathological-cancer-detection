from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def tsne(encoder,loader,device,max_batches,save_dir):
    tsne_h=TSNE(n_components=2,perplexity=30,n_iter=300)
    tsne_z=TSNE(n_components=2,perplexity=30,n_iter=300)

    y_set=torch.tensor([]).to(device)
    h_set=torch.tensor([]).to(device)
    z_set=torch.tensor([]).to(device)

    for k in range(20):
        x,y=next(iter(loader))
        x,y=x.to(device),y.to(device)
        with torch.no_grad():
            h,z=encoder(x)
        y_set=torch.cat([y_set,y])
        h_set=torch.cat([h_set,h])
        z_set=torch.cat([z_set,z])
        print("|",end=" ")

    h_emb=tsne_h.fit_transform(h_set.cpu().numpy())
    z_emb=tsne_z.fit_transform(z_set.cpu().numpy())
    y_set=y_set.cpu().long().numpy()
    plt.subplot(1, 2, 1)
    plt.scatter(h_emb[y_set == 0, 0], h_emb[y_set == 0, 1], label='Class 0')
    plt.scatter(h_emb[y_set == 1, 0], h_emb[y_set == 1, 1], label='Class 1')
    plt.title("h_embeddings")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(z_emb[y_set == 0, 0], z_emb[y_set == 0, 1], label='Class 0')
    plt.scatter(z_emb[y_set == 1, 0], z_emb[y_set == 1, 1], label='Class 1')
    plt.title("z_embeddings")
    plt.legend()
    if save_dir:
        addr=os.path.join(save_dir,"tsne.png")
        plt.savefig(addr)
    plt.show()