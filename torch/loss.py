'''
Description: 
Author: LJJ
Date: 2022-03-27 16:39:52
LastEditTime: 2022-03-27 16:39:53
LastEditors: LJJ
'''
from torch.nn.modules.loss import MSELoss
import torch.nn
from torchvision import vgg19
# import config

class VGGLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.vgg = vgg19(pretrained=True).feature[:36].eval.to(config.DEVICE)
    self.loss = nn.MSELoss()

    for param in self.vgg.parameters():
      param.requires_grad = False

  def forward(self, input, target):
    vgg_input_features = self.vgg(input)
    vgg_target_features = self.vgg(target)
    return self.loss(vgg_input_features, vgg_target_features)
