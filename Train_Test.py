# Train/test measures if a generated model used to predict outcome of certain events is good enough

# Data split into 2 sets
# 80% for training
# 20% for testing

# Train model = create model
# Test model = test accuract of model

# Data
# Illustrates 100 customers in a store, and their shopping habits

import numpy
import matplotlib.pyplot as plt
numpy.random.seed(2)

# random.normal(loc, scale=1.0, size=None)
# draw random samples from a normal (Gaussian) distribution
# loc : float or array of floats
    # mean ("center") of the distribution
# scale : float or array_like of floats
    # standard deviation (spread) of distribution (non negative)
# size : int or tuple of ints, optional
    # output of shape. if given shape is (m, n, k), then m*n*k samples are drawn
    # if size is None, a single value is returned if loc and scale are both scalars
    # otherwise, np.broadcast(loc, scale).size() samples are drawn
x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100)

# x represents number of minutes before making a purchase
# y represents the amount of money spent on a purchase
plt.scatter(x, y)
plt.show()

# Split into train/test (80, 20)

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = x[80:]

# Display same scatter plot with the training set
plt.scatter(train_x, train_y)
plt.show()