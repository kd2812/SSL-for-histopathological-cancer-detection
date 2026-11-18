import torchvision
import torch.nn as nn
# -------------- Encoder Class ----------------
class Encoder(nn.Module):
    """Encoder Model with resnet18 architecture and with last layers formatted. 
        SSLEncoder Use: Used to obtain latent vector projections of images [default dim=128]
        Baseline Use: Used as an MLP with combination with Linear layer 
    """
    def __init__(self,proj_dim=128):
        super().__init__()
        self.encoder=torchvision.models.resnet18(weights=None)
        # last layer formatting
        self.encoder.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.encoder.maxpool=nn.Identity()
        self.encoder.fc=nn.Identity()

        self.projector=nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,proj_dim)
        )
    def forward(self,x):
        h=self.encoder(x)
        z=self.projector(h)
        return h,z

# -------------- Linear Layer Class ----------------
class Linear_Layer(nn.Module):
    """
    Linear layer used to obtain a single logit corresponding to the output.
    SSL Use: used for processing the vector representations further
    Baseline Use: Used as a MLP in addition to the Resnet18 part
    """
    def __init__(self,input_dim=512):
        super().__init__()
        self.ll=nn.Sequential(
            nn.Linear(in_features=input_dim,out_features=64),
            nn.ReLU(),

            nn.Linear(in_features=64,out_features=1),
            # nn.Softmax(dim=1),
            # nn.Sigmoid()
        )
    def forward(self,h):
        return self.ll(h)