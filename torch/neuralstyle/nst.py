'''
Description: 
Author: LJJ
Date: 2022-04-05 16:46:32
LastEditTime: 2022-04-05 17:05:33
LastEditors: LJJ
'''
from pickletools import optimize
from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

class VGG(nn.Module):
  def __init__(self):
    super(VGG,self).__init__()
    
    self.chosen_features = ["0","5","10","19","28"]
    self.model = models.vgg19(pretrained=True).features[:29]
    
  def forward(self,x):
    features = []

    for layer_num, layer in enumerate(self.model):
      x = layer(x)
      if str(layer_num) in self.chosen_features:
        features.append(x)
        
    return features
  
  
def load_image(image_name):
  image = Image.open(image_name)
  image = loader(image).unsqueeze(0)
  return image.to(device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 356

loader = transforms.Compose(
  [
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor(),
  ]
)

original_img = load_image("annahathaway.png")
style_img = load_image("style.jpg")

generated = original_img.clone().requires_grad_(True)
model = VGG().to(device).eval

total_steps = 6000
learning_rate = 0.001
alpha = 1
beta = 0.01
optimizer = optim.Adam([generated], lr=learning_rate)

for step in range(total_steps):
  generated_features = model(generated)
  original_img_features = model(original_img)
  style_features = model(style_img)
  
  style_loss = original_loss = 0
  
  for gen_feature, orig_feature, style_features in zip(
    generated_features, original_img_features,style_features
  ):
    batch_size, channel, height, width = gen_feature.shape
    original_loss += torch.mean((gen_feature - orig_feature) **2)
    G = gen_feature.view(channel,height*width).mm(
      gen_feature.view(channel,height*width).t()
    )
    
    A = style_features.view(channel, height*width).mm(
      style_features.view(channel, height*width).t()
    )
    style_loss += torch.mean((G-A)**2)
    
    total_loss = alpha*original_loss + beta*style_loss
    optimizer.zero_grad*
    total_loss.backward()
    optimizer.step()
    
    if step%200 == 0:
      print(total_loss)
      save_image(generated,"generated.png")
        