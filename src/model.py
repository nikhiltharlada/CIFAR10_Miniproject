import torch
from torch import nn
import torch.nn.functional as F

class CIFARCNN(nn.Module):
    def __init__(self):
        super(CIFARCNN, self).__init__()
        # Layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        
        self.bt1 = nn.BatchNorm2d(32)
        self.bt2 = nn.BatchNorm2d(64)
        self.bt3 = nn.BatchNorm2d(128)
        self.bt4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.dense_layer1 = nn.Linear(256 * 4 * 4, 256)
        self.dense_layer2 = nn.Linear(256, 512)
        self.output = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bt1(self.conv1(x))))
        x = self.pool(F.relu(self.bt2(self.conv2(x))))
        x = self.pool(F.relu(self.bt3(self.conv3(x))))
        x = F.relu(self.bt4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense_layer1(x))
        x = F.relu(self.dense_layer2(x))
        x = self.dropout(x)
        x = self.output(x)
        return x