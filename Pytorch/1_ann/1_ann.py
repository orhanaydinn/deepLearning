# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 15:22:15 2025

Digit classification with Mnist dataset 

@author: Orhan
"""

# %% Library
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch # Tensor 
import torch.nn as nn # Artificial Neural Network for define
import torch.optim as optim # Optimization Algorithm 
import torchvision # Computer Vision Pre-defined 
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt


#optional: Define GPU and CPU 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Data Loading

def get_data_loaders(batch_size = 64): 
    
    transform = transforms.Compose([
                                    transforms.ToTensor(), # Data was converted tensor (0 - 255)
                                    transforms.Normalize((0.5,), (0.5,)) # Pixels value was scaled between -1 and 1
                                    ])
    
    # Download Dataset
    train_set = torchvision.datasets.MNIST(root = "./data", train = True, download=True, transform = transform)
    test_set = torchvision.datasets.MNIST(root = "./data", train = False, download=True, transform = transform)
    
    
    # Upload DataSet
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False)
    
    return train_loader, test_loader
train_loader, test_loader = get_data_loaders()    

# Data Visualization

def visualize_samples(loader, n):
    images, labels = next(iter(loader)) # We can see images and labes from first batch
    fig, axes = plt.subplots(1, n, figsize = (10,5)) # 
    
    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap = "gray")
        axes[i].set_title(f"Labels: {labels[i].item()}")
        axes[i].axis("off")
    plt.show()

visualize_samples(train_loader, 4)

# %% Define ANN Model (Artificial Neural Network)

class NeuralNetwork(nn.Module): 
    
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        
        # The images from MNIST Dataset was converted vector (1D)
        self.flatten = nn.Flatten()
        
        # First Connected  Layer
        self.fc1 = nn.Linear(28*28, 128) # 784 = input size, 128 = output size
        
        # Activation Layer
        self.relu = nn.ReLU()
        
        # Second Fully Connected Layer
        self.fc2 = nn.Linear(128, 64) # 128 = input size, 64 = output size
        
        # Activation Layer
        self.relu = nn.ReLU()
        
        # Output Layer 
        self.fc3 = nn.Linear(64, 10) # 64 = input size, 10 = output size (0-9 labels)

    def forward(self, x): # forward propagation
    
        x = self.flatten(x)  # initial x = 28*28 images
        x = self.fc1(x) # First Connected  Layer
        x = self.relu(x) # Activation Layer
        x = self.fc2(x) # Second Fully Connected Layer
        x = self.relu(x) # Activation Layer
        x = self.fc3(x) # output layer
    
        return x 
            
# Create model and compile

model = NeuralNetwork().to(device)

# Loss Function and Optimization Algo

define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(), # Multi Class classification problems loss function
    optim.Adam(model.parameters(), lr = 0.001) # Update weights with Adam Optim
    )

criterion, optimizer = define_loss_and_optimizer(model)

# %% Train

def train_model(model, train_loader, criterion, optimizer, epochs = 10):
    model.train()
    train_losses = []

    for epoch in range(epochs):
        total_loss = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad() # Reset Grad
            predictions = model(images) # Applied Model
            loss = criterion(predictions, labels) # Calculate Loss
            loss.backward() # Calculate Grad
            optimizer.step() # Update Weights

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, loss: {avg_loss:.3f}")

    # loss graph (döngü bittikten sonra çiz)
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, marker = "o", linestyle = "-", label = "Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()

train_model(model, train_loader, criterion, optimizer, epochs = 5)
# %% Test 

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0 # total data value
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            _, predicted = torch.max(predictions, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"Test Accuracy: {100*correct/total:.3f}%")

test_model(model, test_loader)

# %%main

if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    visualize_samples(train_loader, 5)
    model = NeuralNetwork().to(device)
    criterion, optimizer = define_loss_and_optimizer(model)
    train_model(model, train_loader, criterion, optimizer)
    test_model(model, test_loader)
