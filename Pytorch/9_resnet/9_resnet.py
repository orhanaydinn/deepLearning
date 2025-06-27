"""
Resnet siniflandirma -> CIFAR10 
    - transfer learning
    - custom resnet build
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models 
from tqdm import tqdm 

# %% Upload Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Upload Data
transform = transforms.Compose([
    transforms.ToTensor(), # gorselleri tensore cevir
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# CIFAR10 Dataset
trainset = torchvision.datasets.CIFAR10(root = "./data", train = True, download=True, transform = transform)
testset = torchvision.datasets.CIFAR10(root = "./data", train = False, download = True, transform = transform)

# data loader
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# %% Residual block

class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        
        """
        cov2d -> batchNorm -> relu -> con2d -> batchNorm -> downsampling 
        """
        
        super(ResidualBlock, self).__init__()
        
        # 3x3 conv2d
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride = stride, padding = 1, bias = False)
        
        # batch normalization Layer
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # relu activation
        self.relu = nn.ReLU()
        
        # 3x3 conv2d
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        
        # batch normalization layer
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # downsampling
        self.downsample = downsample
    
    def forward(self, x):
        
        identity = x # connected layer info
        
        if self.downsample is not None:
            identity = self.downsample(x) 
            
        out = self.conv1(x) # First Conv 
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity # skip connection
        out = self.relu(out)
        
        return out
        
# %% Create ResNet custom

class CustomResNet(nn.Module):
    
    def __init__(self, num_classes = 10):
        """
            conv2d -> batchNorm -> relu -> maxpool -> 4 x layer -> avgpool -> fc
        """
        super(CustomResNet, self).__init__()
        
        # Fist conv
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding  = 3, bias = False)
        
        # batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        
        # relu
        self.relu = nn.ReLU()
        
        # max pooling
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 4 x make layer
        self.layer1 = self._make_layer(64,64,2) # 64 kanalli ilk katman
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, 512, 2, 2)
        
        # avg pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        # Fully connected layer
        self.fc = nn.Linear(512, num_classes)
        
    
    def _make_layer(self, in_channels, out_channels, blocks, stride = 1): # residual katmanlari olusturan fonksiyon
        downsample = None 
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels)
                )
        
        # First residual block
        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]
        # Other residual blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        x = self.conv1(x) # First conv
        x = self.bn1(x) # batch normalization
        x = self.relu(x) # Activation
        x = self.maxpool(x) # pooling
        
     
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x) # pooling
        x = torch.flatten(x, 1)
        x = self.fc(x) # Fulley connected layer
        return x
        
# model = CustomResNet()
# %% Training 
use_custom_model = True #
if use_custom_model:
    model = CustomResNet().to(device) 
else:
    model = models.resnet18(pretrained = True) #
    num_ftrs = model.fc.in_features # 
    model.fc = nn.Sequential( 
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Linear(256, 10)) # output layer
    model = model.to(device)
    
# loss function and optimizer
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Training
num_epochs = 1
for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss = 0
    
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad() 
        outputs = model(images) 
        loss = criterion(outputs, labels) # loss calculate
        loss.backward()  
        optimizer.step()  # weights update
        running_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{num_epochs}, loss: {running_loss/len(train_loader)}")

# %% test and evaluation

model.eval()
correct = 0 
total = 0 
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test accuracy: {100 * correct / total}%")

