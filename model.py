import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # First conv layer: 1 input channel (grayscale), 8 output channels
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1)
        # Second conv layer: 8 input channels, 16 output channels
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.2)
        # Calculate the size for the fully connected layer
        # After 2 conv layers without padding and 2 max pools, 
        # the 28x28 image becomes 5x5
        self.fc = nn.Linear(32 * 5 * 5, 10)
        
    def forward(self, x):
        # First conv block
        x = self.conv1(x) # o/p : 26x26x1
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # o/p : 13x13x1
        
        # Second conv block
        x = self.conv2(x) # o/p: 13x13x1
        x = F.relu(x)
        x = self.conv3(x) # o/p: 11x11x1
        x = F.relu(x)
        x = F.max_pool2d(x, 2) #o/p: 5x5x1
        x = self.dropout(x)
        
        # Flatten and pass through fully connected layer
        x = x.view(-1, 32 * 5 * 5)
        x = self.fc(x)
        return x