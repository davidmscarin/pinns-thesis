import torch
import numpy as np


#neural net with 3 fully connected layers and ReLU activation
class model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.relu = torch.nn.ReLU
        self.fc1 = torch.nn.Linear(1, size)
        self.fc2 = torch.nn.Linear(size, size)
        self.out = torch.nn.Linear(size, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.out(x)


