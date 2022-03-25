## Automatic differentiation engine that powers neural network training

# In this ex. we load a pretrained resnet18 model from torchvision
# Create a random tensor to represent a single image with 3 channels, height & width of 64
# and corresponding label initialized to some random value
# Label in pretrained models has shape (1, 1000)

import torch, torchvision
model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# Run the input data through the model through each layer to make prediction (forward pass)
prediction = model(data) # forward pass

# Use model's prediction and corresponding label to calculate the error (loss)
# Backpropogate this error through the network
# .backward() backpropogates, then autograd calculates and stores the gradients for each model
# parameter in the parameters .grad attribute
loss = (prediction - labels).sum()
loss.backward() # backward pass

# Load optimizer (SGD in this case) w/ a learning rate of 0.01 and momentum of .9
# Register all params of model in the optimizer
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# Initiate gradient descent, optimizer adjusts each parameter by its gradient stored in .grad
optim.step()

