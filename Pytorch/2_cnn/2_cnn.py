# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 15:38:08 2025

@author: Orhan
"""

# %% import libraries

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt  
import numpy as np

#optional: Define GPU and CPU 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load dataset
def get_data_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # parantez düzeltildi
    ])

    # Download CIFAR10 DataSet
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# %% Visualize Dataset

def imshow(img):
    img = img /2 + 0.5 #Normalize
    np_img = img.numpy() 
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()
    
def get_sample_images(train_loader):
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    return images, labels

def visualize(n):
    train_loader, test_loader = get_data_loaders()
    
    images, labels = get_sample_images(train_loader)
    plt.figure()
    for i in range(n):
        plt.subplot(1, n, i+1)
        imshow(images[i])
        plt.title(f"Label: {labels[i].item()}")
        plt.axis("off")
    plt.show()
    
visualize(10)

# %% build CNN Model

class CNN(nn.Module):
    
    def __init__(self):
        
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2) # 2x2 pooling layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.dropout = nn.Dropout(0.2) #dropout %20
        self.fc1 = nn.Linear(64*8*8, 128) #fully connected layer, input = 4096, output = 128
        self.fc2 = nn.Linear(128, 10)
        
        
        # image 3x32x32 -> conv(32) -> relu(32) -> pool(16)
        # conv(16) -> relu(16) -> pool(8) -> image 8x8

        
    def forward(self, x):
        """
        image 3x32x32 -> conv(32) -> relu(32) -> pool(16)
        conv(16) -> relu(16) -> pool(8) -> image 8x8
        flatten
        fc1 -> relu -> dropout
        fc2 -> output
        """
        x = self.pool(self.relu(self.conv1(x))) # First convolution Block
        x = self.pool(self.relu(self.conv2(x))) # Second convolution Block
        x = x.view(-1, 64*8*8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = CNN().to(device)

#define loss function and optimizer

define_loss_and_optimizer = lambda model: (
                                            nn.CrossEntropyLoss(),
                                            optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9) # Stochastic Gradient Descent
                                            )
# %% Training

def train_model(model, train_loader, criterion, optimizer, epochs = 5):
    
    # Start Training Model
    model.train()
    
    # Create a list for keeping loss values
    train_losses = []
    
    # Create a for structer according to epochs
    for epoch in range(epochs):
        total_loss = 0 
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
           
            # Reset Gradient 
            optimizer.zero_grad()
            # Forward Pro. (Prediction)
            outputs = model(images)
            # Calculate Loss Value
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
          
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.5f}")
        
        
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, marker = "o", linestyle = "-", label = "Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()

train_loaders, test_loaders = get_data_loaders()
model = CNN().to(device)
criterion, optimizer = define_loss_and_optimizer(model)
train_model(model, train_loaders, criterion, optimizer, epochs = 5)
    

# %% Test

def test_model(model, test_loaders, dataset_type):
    
    model.eval()
    correct = 0 
    total = 0
    
    with torch.no_grad(): #stop gradient calculation
        for images, labels in test_loaders:
            images,labels = images.to(device), labels.to(device)
            
            outputs = model(images) # prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"Test Acc: {100* correct / total}%")

test_model(model, test_loaders, dataset_type = "test")
test_model(model, train_loaders, dataset_type = "training")

    