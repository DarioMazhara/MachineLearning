import numpy as np
import matplotlib.pyplot as plt

# 3d plot
fig = plt.figure()
ax = plt.axes(projection='3d')

# Plotting 3d lines and plots
# 3 dimensional line graph
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')

# def all 3 axes
z = np.linspace(0, 1, 100)
x = z * np.sin(25 * z)
y = z * np.cos(25 * z)

ax.plot(x, y, z, 'green')
ax.set_title('3d line plot')


# 3d scattered graph

c = x + y
ax.scatter(x, y, z, c = c)
ax.set_title('3d scatter plot')
plt.show()