

from statistics import LinearRegression
from numpy import DataSource
from sklearn import datasets, linear_model
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.autograd
import torch.nn
# Five steps in the life-cycle

# 1. Prepare the data
# 2. Define the model 
# 3. Train the model 
# 4. Evaluate the model 
# 5. Make predictions 

### Step 1: Prepare the data

# Can use std python libs. to load & prepare tabular data (like CSV)
# PyTorch has Dataset class that you can extend & use to customize your dataset

# Ex: construct, can load your data file
# __len__() can get length of dataset (#rows or #samples)

# When loading dataset, can perform required transformations like scaling
# or encoding

# Custom Dataset class

# dataset definition
class CSVDataset(Dataset):
    # load dataset
    def __init__(self, path):
        # store inps and outps
        self.X = ...
        self.y = ...
    
    # number of rows in dataset
    def __len__(self):
        return len(self.X)

    # get row by index
    def __getitem__(self, index):
        return [self.X[index], self.y[index]]
    
# Use PyTorch's DataLoader to navigate a Dataset instance
    # DataLoader instance can be made for any set (training, testing, validation)

# random_split() can split dataset into train and test sets
    # selection of rows from Dataset can be given to DataLoader
        # and batch size and if data should be shuffled each epoch
    
    
# create dataset
dataset = CSVDataset(...)
# select rows from dataset
train, test = random_split(dataset, [[...], [...]])
# create data loader for train & test
train_dl = DataLoader(train, batch_size=32, shuffle=True)
test_dl = DataLoader(test, batch_size=32, shuffle=False)

# after def. DataLoader can be enumerated, yielding one batch of samples
# per iteration

# train model
for i, (inputs, targets) in enumerate(train_dl):
    ...


### Step 2: Define the Model

# Defining a model in PyTorch involves defining a class that
# extends the Module class
    # Construt. defines layers 
    # forward() overrides definition for how to propogate input
        # through defined layers
        
    # Many layers available
        # Linear for fully connected layers
        # Conv2d for convolutional layers
        # MaxPool2d for pooling layers
        # Activation functions can also be layers (ReLU, Softmax, Sigmoid)

# MLP model w/ one layer

# model def
def MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.layer = Linear(n_inputs, 1)
        
