# Tensors used to enconde inputs and outputs of a model, and models params

# Tensors can run on GPU to accelerate computing 

import torch
import numpy as np

## Tensor initilization
# Directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# From numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# From other tensor
x_ones = torch.ones_like(x_data)
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides datatype of x_data
# print(f"Random tensor: \n {x_rand} \n")

# With random or const values
# shape is a tuple of tensor dimensions
# Below determines the dimensionality of the output tensor
shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zero_tensor = torch.zeros(shape)

## Tensor attributes
# describe their shape, datatype, and device they're on
tensor = torch.rand(3, 4)
print(tensor.shape)
print(tensor.dtype)
print(tensor.device)

## Tensor operations
# Move tensor to gpu if possible
if torch.cuda_is_available():
    tensor = tensor.to('cuda')
    
# Indexing and slicing
tensor = torch.ones(4, 4)
tensor[:,1] = 0
# Output of tensor
# tensor([[1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.]])

# Joining tensors
# Concat seq of tensors along a given dim
t1 = torch.cat([tensor, tensor, tensor], dim = 1)

# Multiplying tensors
# element wise produce
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Or
print(f"tensor * tensor \n {tensor * tensor} \n")

# Matrix mul between 2 tensors
tensor.matmul(tensor.T)
# or
tensor @ tensor.T


