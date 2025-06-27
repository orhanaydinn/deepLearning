# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 16:26:51 2025

@author: Orhan
"""
"""
Dataset: FashionMNIST
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np

#  Upload Dataset 

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.FashionMNIST(root = "./data", train = True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root = "./data", train = False, transform=transform, download=True)


batch_size = 256

train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle = False)

#  Develop Autoencoders

class AutoEncoder(nn.Module):
    
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        # encoder
        self.encoder = nn.Sequential(
            nn.Flatten(), # 28x28 -> 784 vector
            nn.Linear(28*28, 256), # Fully connected layer
            nn.ReLU(), # Activation Func
            nn.Linear(256, 64), # Fully connected layer
            nn.ReLU() # Activation Func 
            )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 256), # Fully connected layer
            nn.ReLU(), # Activation Func
            nn.Linear(256, 28*28), # Fully connected layer
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
            )
        
    def forward(self, x):
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
#  Callback: early stopping

class EarlyStopping: # Callback class
    
    def __init__(self, patience = 5, min_delta = 0.001):
        
        self.patience = patience 
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        
        
    def __call__(self, loss):
        
        if self.best_loss is None or loss < self.best_loss - self.min_delta: 
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter +=1
            
        if self.counter >= self.patience:
            return True
        
        return False
        
#  Model Training

# Hyperparameters

epochs = 50
learning_rate = 1e-3

# Def Model

model = AutoEncoder()
criterion = nn.MSELoss() # Loss Func
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
early_stopping = EarlyStopping(patience=3, min_delta=0.001) # callback

def training(model, train_loader, optimizer, criterion, early_stopping, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss/len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, loss: {avg_loss:.5f}")

        # Early Stopping
        if early_stopping(avg_loss):
            print("Early Stopping at epoch {epoch+1}")
            break
        
training(model, train_loader, optimizer, criterion, early_stopping, epochs)

# %%  Model Testing

from scipy.ndimage import gaussian_filter

def compute_ssim(img1, img2, sigma=1.5):
    
    """
    Calculation between two images similarity
    
    """
    C1 =(0.01*255)*2
    C2 =(0.03*255)*2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mu1 = gaussian_filter(img1, sigma)
    mu2 = gaussian_filter(img2, sigma)
    mu1_mu2 = mu1* mu2


    mu1_sq = mu1**2
    mu2_sq = mu2**2 
    
    sigma1_sq = gaussian_filter(img1 **2, sigma) - mu1_sq    
    sigma2_sq = gaussian_filter(img2 **2, sigma) - mu2_sq    
    sigma12 = gaussian_filter(img1*img2, sigma) - mu1_mu2
    
    ssim_map = ((2*mu1_mu2 + C1)* (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    return ssim_map


def evaluate(model, test_loader, n_images = 10):
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            inputs, _ = batch
            outputs = model(inputs)
            
    inputs = inputs.numpy()
    outputs = outputs.numpy()
    
    fig, axes = plt.subplots(2, n_images, figsize = (n_images, 3))
    
    ssim_scores = []  # Bu satır eksikti
    
    for i in range(n_images):
        img1 = np.squeeze(inputs[i])
        img2 = np.squeeze(outputs[i])
    
        score = compute_ssim(img1, img2)  # tekil skor
        ssim_scores.append(score)         # listeye ekle
    
        axes[0,i].imshow(img1, cmap="gray")
        axes[0,i].axis("off")
        axes[1,i].imshow(img2, cmap="gray")
        axes[1,i].axis("off")
        axes[0,0].set_title("Original")
        axes[1,0].set_title("Decoded Image")
    
    avg_ssim = np.mean(ssim_scores)
    print(f"Average SSIM: {avg_ssim}")
    
evaluate(model, test_loader, n_images = 10)









