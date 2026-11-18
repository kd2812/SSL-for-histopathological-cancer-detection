import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from src.data_augmentations import denorm
class GradCam():
  def __init__(self,encoder,final_layer,target_layer):
    self.encoder=encoder
    self.final_layer=final_layer
    self.device = next(encoder.parameters()).device
    self.target_layer=target_layer
    self.gradients=None
    self.activations=None

    self.encoder.zero_grad()
    self.final_layer.zero_grad()
    self.encoder.requires_grad_(True)
    self.final_layer.requires_grad_(True)
    self.target_layer.register_forward_hook(self.save_activations)
    self.target_layer.register_backward_hook(self.save_grad)
  def forward(self,x):
    x1,_=self.encoder(x)
    x2=self.final_layer(x1)
    return x2


  def save_grad(self,module,grad_input,grad_output):
    self.gradients=grad_output[0]


  def save_activations(self,module,input,output):
    self.activations=output

  def generate(self,image,class_idx=None):

    image=image.to(self.device)
    self.encoder.zero_grad()
    self.final_layer.zero_grad()
    output=self.forward(image)
    logit = output[:, 0]

    if class_idx is None:
        class_idx = (logit >= 0).long().item()


    score = logit if class_idx == 1 else -logit

    score.sum().backward()
    actv=self.activations
    grad=self.gradients

    weights=grad.mean(dim=(2,3),keepdim=True)

    cam=(weights*actv).sum(dim=1)
    cam=F.relu(cam)
    cam=cam-cam.min()
    cam=cam/(cam.max()+1e-8)
    cam=F.interpolate(cam.unsqueeze(0),size=image.shape[2:],mode='bilinear',align_corners=False)
    return cam.squeeze().detach(),class_idx




def show_gradcam(gradcam,x,y,alpha=0.3,save=False,dir=None,title=None):



  cam,class_idx=gradcam.generate(x,class_idx=y.item())
  image=denorm(x.cpu().squeeze()).permute(1,2,0)
  cam=cam.cpu().numpy()
  cam_colored=plt.cm.jet(cam)[...,:3]
  image_norm=(image/255.0)
  image_with_cam=image*alpha+cam_colored*(1-alpha)
  plt.subplot(1,2,1)
  plt.imshow(image)
  plt.title(f"Image y={class_idx}")
  plt.subplot(1,2,2)
  plt.imshow(image_with_cam)
  plt.title("Grad-CAM")
  plt.show()
  if save:
    if dir is not None:
      dir=os.path.join(dir,f"{title}.png")
      plt.imsave(dir,image_with_cam)
      print("image saved")
    else:
      print("dir is not specified")