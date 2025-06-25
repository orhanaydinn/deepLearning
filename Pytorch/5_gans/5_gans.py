# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 16:08:16 2025

@author: Orhan
"""

"""

Image Generation: MNIST DataSet

"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Dataset Preperation 

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as utils
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt 
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
image_size = 28*28

transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)) # Normalization -> -1 and 1
                                ])

# MNIST DataSet Upload

dataset = datasets.MNIST(root = "./data", train = True, transform=transform, download=True)


# Dataset was upload with batch
dataLoader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

# Create Discriminator 

class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
                                   nn.Linear(image_size, 1024), #|input : image size, 1024
                                   nn.LeakyReLU(0.2), # Activation func.
                                   nn.Linear(1024, 512),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(512, 256),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(256, 1), # 256 -> 1
                                   nn.Sigmoid() # 0-1
                                   )                              
        
    def forward(self, img):
        return self.model(img.view(-1, image_size))

# Create Generator

class Generator(nn.Module):
    
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256), #input to 256 fully connected layer
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024, image_size), # 1024 to 784 transform
            nn.Tanh() # Output Layer
            )
    
    def forward(self, x):
        return self.model(x).view(-1, 1, 28, 28)
    
# GAN Training

# Hyperparameter
learning_rate = 0.0002 # Learning Rate
z_dim = 100 # Random Noise
epochs = 50 # Training Loop




# Start Models: Describe generator and discriminator 
generator = Generator(z_dim).to(device)
discriminator = Discriminator().to(device)


# Describe Loss Func and Optimization 
criterion = nn.BCELoss() # Binary Cross Entropy
g_optimizer = optim.Adam(generator.parameters(), lr = learning_rate, betas = (0.5, 0.999)) # generator
d_optimizer = optim.Adam(discriminator.parameters(), lr= learning_rate, betas = (0.5, 0.999)) # discriminator


print("Training Device:", device)
print("Generator Device:", next(generator.parameters()).device)


# Start Training Loop
for epoch in range(epochs):
    for i, (real_imgs, _) in enumerate(dataLoader): #☺ upload image
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)
        real_labels = torch.ones(batch_size, 1).to(device) # real image label : 1
        fake_labels = torch.zeros(batch_size, 1).to(device) # Fake image label : 0
        
        
        # Discriminator Training
        z = torch.randn(batch_size, z_dim).to(device) # produce random image
        fake_imgs = generator(z)
        real_loss = criterion(discriminator(real_imgs), real_labels) # Real Image Loss
        fake_loss = criterion(discriminator(fake_imgs.detach()),fake_labels) # Fake Image Loss
        d_loss = real_loss + fake_loss # Total Discriminator Loss
        
        d_optimizer.zero_grad() # Reset Grad
        d_loss.backward()
        d_optimizer.step()
        
        
        # Generator Training
        g_loss = criterion(discriminator(fake_imgs), real_labels) # Generator Labels
        g_optimizer.zero_grad() # Reset Grad
        g_loss.backward() # Backward
        g_optimizer.step() # Update Parameters
    
    print(f"Epoch {epoch +1}/{epochs} d_loss: {d_loss.item():.3f}, g_Loss: {g_loss.item():.3f}")
print("Training was finished...")
# Model Testing



with torch.no_grad():
    z = torch.randn(16, z_dim).to(device)
    sample_imgs = generator(z).cpu()
    grid = np.transpose(utils.make_grid(sample_imgs, nrow=4, normalize=True), (1,2,0))
    plt.imshow(grid)
    plt.show()
