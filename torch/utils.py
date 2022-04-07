'''
Description: 
Author: LJJ
Date: 2022-03-27 16:42:42
LastEditTime: 2022-03-27 22:29:39
LastEditors: LJJ
'''
from pickletools import optimize
import torch
import os 
import config 
import numpy as np
from PIL import Image
from torchvision import save_image


def gradient_penalty(critic, real, fake, device):
  BATCH_SIZE, C, H, W = real.shape
  alpha = torch.rand(BATCH_SIZE, 1,1,1).repeat(1,C,H,W).to(device)
  interpolated_images = real*alpha+fake.detach()*(1-alpha)
  interpolated_images.required_grad_(True)
  
  mixed_scores = critic(interpolated_images)
  
  gradient = torch.autograd.grad(
    inputs=interpolated_images,
    outputs=mixed_scores,
    grad_outputs=torch.ones_like(mixed_scores),
    create_graph=True,
    retain_graph=True,
  )[0]
  gradient = gradient.view(gradient.shape0[0],-1)
  gradient_norm = gradient.norm(2,dim=1)
  gradient_penalty = torch.mean((gradient_norm-1)**2)
  return gradient_penalty

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
  print("->Saving checkpoint")
  checkpoint = {
    "state_dic" : model.state_dict(),
    "optimizer" : optimizer.state_dict(),
  }
  torch.save(checkpoint,filename)
  
  
def load_checkpoint(checkpoint_file, model, optimizer, lr):
  print("=>loading checkpoint")
  checkpoint = torch.load(checkpoint_file,map_location=config.DEVICE)
  model.load_state_dict(checkpoint["state_dict"])
  optimizer.load_state_dict(checkpoint["optimizer"])
  
  for param_group in optimizer.param_groups:
    param_group["lr"] = lr
  
def plot_examples(low_res_folder,gen):
    files = os.listdir(low_res_folder)
    
    gen.eval()
    for file in files:
      image = Image.open("test_images/" + file)
      with torch.no_grad():
        upscaled_img = gen(
          config.test_transform(image=np.asarray(image))["image"]
          .unsqueeze(0)
          .to(config.DEVICE)
        )
        save_image(upscaled_img*0.5+0.5,f"save/{file}")
        
    gen.train()