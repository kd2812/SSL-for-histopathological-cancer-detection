import torch 
import torch.nn.functional as F

# ----------- NTXENT Loss Function -----------------

def NTXent_Loss(z,tau):
    """
    z:[2N,d]

    returns the NTXent Loss for a given batch 
    
    """
    n2=z.shape[0]
    n=n2//2

    z=F.normalize(z,dim=-1)
    sim=z @ z.T /tau

    pos=torch.arange(n2,device=z.device)

    positive_pos=pos^1

    mask=torch.eye(n2, dtype=torch.bool,device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    log_soft=F.log_softmax(sim,dim=-1)
    loss=-log_soft[pos,positive_pos].mean()

    return loss