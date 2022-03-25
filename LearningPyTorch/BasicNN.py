### Neural Network

import torch
import torch.nn as nn
# Simple neural network with one hidden layer developing
# a single output unit


# Define input size, hidden layers, output size, and batch size
# 10 inputs, 1 hidden layer with 5 nodes, 1 output node, batch size
n_in, n_h, n_out, batch_size = 10, 5, 1, 10

# Create dummy input and target tensors
x = torch.randn(batch_size, n_in)
y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])

# Create a model
model = nn.Sequential(nn.Linear(n_in, n_h),
    nn.ReLU(),
    nn.Linear(n_h, n_out),
    nn.Sigmoid())

# Construct loss function with help of gradient descent optimizer
criterion = torch.nn.MSELoss()
# Construct the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# Gradient Descent

for epoch in range(50):
    # Forward pass: computed predicted y by passing x to the model
    y_pred = model(x)
    
    # Compute and print loss
    loss = criterion(y_pred, y)
    print('epoch: ', epoch, ' loss: ', loss.item())
    
    # Zero gradients, perform a backward pass, update weights
    optimizer.zero_grad()
    
    # Perform a backward pass (backpropogation)
    loss.backward()
    
    # Update the parameters
    optimizer.step()