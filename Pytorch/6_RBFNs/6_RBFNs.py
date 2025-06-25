# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 15:45:20 2025

@author: Orhan
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# %% Import DataSet 

"""
Classification Problem with Iris Dataset 
"""

df = pd.read_csv("C:/Users/Orhan/Desktop/Software Project/Artificial Intelligence/Deep Learning Tecniques/Pytorch/6_RBFNs/iris.data/iris.data", header = None)

X = df.iloc[:, :-1].values
y, _ = pd.factorize(df.iloc[:, -1])

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

def to_tensor(data, target):
    return torch.tensor(data, dtype = torch.float32), torch.tensor(target, dtype = torch.long)

X_train, y_train = to_tensor(X_train, y_train)
X_test, y_test = to_tensor(X_test, y_test)


# %% RBFN Model and Describe RBF_Kernel

def rbf_kernel(X, centers, beta):
    return torch.exp(-beta * torch.cdist(X, centers)**2)

class RBFN(nn.Module):
    
    
    def __init__(self, num_centers, input_dim, output_dim):
        super(RBFN, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, input_dim)) # Randomly start rbfm centers
        self.beta = nn.Parameter(torch.ones(1)*2.0) # beta parameter will control rbf width
        self.linear = nn.Linear(num_centers, output_dim) # directs output to fully connected layer
        
        
    def forward(self, x):
        # Calculate RFB func
        phi = rbf_kernel(x, self.centers, self.beta)
        return self.linear(phi)
        
    
#model = RBFN(4, 10,3)
# %% Model Training

num_centers = 10
model = RBFN(input_dim=4, num_centers=num_centers, output_dim=3)

# Describe Loss func
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.002)

#Training Model

num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if(epoch +1) % 10 == 0:
        print(f"Epoch {epoch +1}/{num_epochs}, Loss: {loss.item():.4f}")
    

# %% Test and Evaluation 

with torch.no_grad():
    y_pred = model(X_test)
    accuracy = (torch.argmax(y_pred, axis = 1) == y_test).float().mean().item()
    print(f"Acc: {accuracy}")







