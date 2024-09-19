import numpy as np

data = np.array([1, 20, 13, 14, 15])

beta = 0.1

p = np.exp(-beta * data)

print(p)

