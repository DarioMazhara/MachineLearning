### Convolutional Neural Network

# Designed to process data through multiple layers of arrays
# This type of NN used in things like image or face recognition

# Difference between Convolutional and ordinary NN, 
# CNN takes input as a two dimensional array and operates
# directly on the images rather than focusing on feature extraction


# Every CNN has 3 basic ideas
## Local respective fields
## Convolution
## Pooling

# Local respective fields = region in which an individual hidden neuron
# is processing a select amount of input data oblivious to changes outside the field

# Convolution = each connection learns weight of hidden neuron w/ an
# association connection w/ movement from one layer to another
#  individual neruons will occasionally perform a shift (convolution) 

import torch
from torch.autograd import Variable
import torch.nn.functional as F

# Class with batch representation of CNN.
# Batch shape = x, dimensions = (3, 32, 32)
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Inp channels = 3, outp channels = 18
        self.convl = torch.nn.Conv2d(3, 18, kernel_size = 3, stride = 1, padding = 1)
        self.pool = torch.nn.MaxPool2d(kernel_size= 2, stride = 2, padding = 0)
        # 4608 inp features, 64 outp features
        self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
        # 64 inp feats, 10 outp feats for 10 defined classes
        self.fc2 = torch.nn.Linear(64, 10)
        
        
    def forward(self, x):
        x = F.relu(self.convl(x))
        x = self.pool(x)
        x = x.view(-1, 18 * 16 * 16)
        x = F.relu(self.fc1(x))
        # Computes second fully connected layer(activation applied later)
        # Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        return (x)



def main():
    module = torch.nn.Module
    scnn = SimpleCNN()
    x = torch.tensor([[1, 1], [3, 5]])
    
    print (scnn.forward(x))
    
if __name__=="__main__":
    main()