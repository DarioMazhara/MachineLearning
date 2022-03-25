# Neural network depends on autograd to define models and differentiate them
# nn.Module contains layers and forward(input) = output

# Simple feed-forward network takes input, feeds thru several layers one by one, then gives output

# NN training
    # Def nn with learnable params
    # Iterate over inputs
    # Process input thru network
    # Compute the loss
    # Propogate back gradients into the params
    # Updates weights 
    
import time
from time import sleep
from sinchsms import SinchSMS
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    # Define network w/ learnable params
    def __init__(self):
        super(Network, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # affine operation: y=Wx+b
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 5x5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # Max pooling over (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If size is square, can specify with single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except batch dimension
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Network()
print(net)

# Learnable params of model returned by net.parameters()
params = list(net.parameters())
print(len(params))
print(params[0].size()) #conv1's weight

# Random 32x32 input
input = torch.rand(1, 1, 32, 32)
out = net(input)
print(out)