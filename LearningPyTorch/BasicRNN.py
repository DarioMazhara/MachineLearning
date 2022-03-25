### Recurrent Neural Network

# A type of deeplearning oriented algorithm
# Follows sequential approach
    # perform math operations in a seq. manner (one after another)
    
### Creating sine wave with recurrent neural networks

## Training approach to model with one data point at a time
## input seq. x has 20 data points
## target seq = same as input seq

from matplotlib.style import context
from numpy import require
import torch
from torch.autograd import Variable
import numpy as np
import pylab as pl
import torch.nn.init as init

# Seet model hyperparameters w/ size of inp layer = 7
# 6 context neurons, 1 input neuron for creating target seq

dtype = torch.FloatTensor
input_size, hidden_size, output_size = 7, 6, 1
epochs = 300
seq_length = 20
lr = 0.1
data_time_steps = np.linspace(2, 10, seq_length + 1)
data = np.sin(data_time_steps)
data.resize((seq_length + 1, 1))

x = Variable(torch.Tensor(data[:-1]).type(dtype), requires_grad=False)
y = Variable(torch.Tensor(data[1:]).type(dtype), requires_grad=False)
# Generate training data, where x is in input data seq and y is required target seq

# Weights init. in RNN using normal distribution w/ zero mean
# W1 = acceptance of inp. vars
# W2 = generated output

w1 = torch.FloatTensor(input_size, hidden_size).type(dtype)
init.normal(w1, 0,4)
w1 = Variable(w1, requires_grad=True)
w2= torch.FloatTensor(hidden_size, output_size).type(dtype)
init.normal(0.0, 0.3)
w2 = Variable(w2, requires_grad=True)

# Function for feed forward, uniquely defines the network

def forward(input, context_state, w1, w2):
    xh = torch.cat((input, context_state), 1)
    context_state = torch.tanh(xh.mm(w1))
    out = context_state.mm(w2)
    return (out, context_state)

# Training procedure of RNN's sine wave implementation
    # Outer loop iterates over each loop
    # Inner loop iterates through elements of sequence
    # Compute Mean Square Error (MSE), helps predict continouous variables

for i in range(epochs):
    total_loss = 0
    context_state = Variable(torch.zeros((1, hidden_size)).type(dtype), requires_grad=True)
    # Iterates through elements of sequence
    for j in range(x.size(0)):
        input = x[j:(j+1)]
        target = y[j:(j+1)]
        (pred, context_state) = forward(input, context_state, w1, w2)
        loss = (pred-target).pow(2).sum()/2
        total_loss += loss
        loss.backward()
        w1.data -= lr * w1.grad.data
        w2.data -= lr * w2.grad.data
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        context_state = Variable(context_state.data)
    if i % 10 == 0:
        print ("Epoch: {} loss {}".format(i, total_loss.data[0]))

context_state = Variable(torch.zeros((1, hidden_size)).type(dtype), requires_grad=False)
predictions = []

for i in range(x.size(0)):
    input = x[i:i+1]
    (pred, context_state) = forward(input, context_state, w1, w2)
    context_state = context_state
    predictions.append(pred.data.numpy().ravel()[0])
    
# Plot sine wave
pl.scatter(data_time_steps[:-1], x.data.numpy(), s = 90, label = "Actual")
pl.scatter(data_time_steps[1:], predictions, label = "Predicted")
pl.legend()
pl.show()