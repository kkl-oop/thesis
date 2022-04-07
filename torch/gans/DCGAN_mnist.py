'''
Description: 
Author: LJJ
Date: 2022-03-29 14:11:42
LastEditTime: 2022-03-29 15:50:38
LastEditors: LJJ
'''

from cv2 import normalize
from numpy import reshape
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import (
  DataLoader,
)
from torch.utils.tensorboard import SummaryWriter
from model_utils import (
  Discriminator,
  Generator,
)

lr = 0.0005
batch_size = 64
image_size = 64
channels_img = 1
channels_noise = 256
num_epochs = 10

features_d = 16
features_g = 16

my_transforms = transforms.Compose(
  [
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)),
  ]
)

dataset = datasets.MNIST(
  root="dataset/", train=True, transform=my_transforms, download=True,
)

dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

netD = Discriminator(channels_img, features_d,).to(device)
netG = Generator(channels_noise, channels_img, features_g).to(device)

optimizerD = optim.Adam(netD.parameters(),lr=lr,betas=(0.5,0.999))
optimizerG = optim.Adam(netG.parameters(),lr=lr,betas=(0.5,0.999))

netD.train()
netG.train()

criterion = nn.BCELoss()

real_label = 1
fake_label = 0

fixed_noise = torch.randn(64,channels_noise,1,1).to(device)
writer_real = SummaryWriter(f"runs/GAN_MNIST/test_real")
writer_fake = SummaryWriter(f"runs/GAN_MNIST/test_fake")
step = 0

print("Starting Training...")

for epoch in range(num_epochs):
  for batch_idx, (data, targets) in enumerate(dataloader):
    data = data.to(device)
    batch_size = data.shape[0]
    
    netD.zero_grad()
    label = (torch.ones(batch_size)*9).to(device)
    output = netD(data).reshape(-1)
    lossD_real = criterion(output, label)
    D_x = output.mean().item()
    
    noise = torch.randn(batch_size, channels_noise, 1,1).to(device)
    fake = netG(noise)
    label = (torch.ones(batch_size)*0.1).to(device)
    
    output = netD(fake.detach()).reshape(-1)
    lossD_fake = criterion(output,label)
    
    lossD = lossD_real + lossD_fake
    lossD.backward()
    optimizerD.step()
    
    netG.zero_grad()
    label = torch.ones(batch_size).to(device)
    output = netD(fake).reshape(-1)
    lossG = criterion(output,label)
    lossG.backward()
    optimizerG.step()
    
    if batch_idx %100 == 0:
      step += 1
      print(
        f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} Loss D: {lossD:.4f}, loss G: {lossG:.4F} D(x): {D_x:.4f}"
      )
      
      with torch.no_grad():
        fake = netG(fixed_noise)
        img_grid_real = torchvision.utils.make_grad(data[:32], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:32],normailize= True)
        writer_real.add_image(
          "Mnist Real Images", img_grid_real, global_step=step,
        )
        writer_fake.add_image(
          "Mnist Fake Images", img_grid_fake, global_step=step,
        )