import numpy as np

x=np.zeros((2,3,4))

input_shape= x.shape[0] * x.shape[1] * x.shape[2]
input_shape = np.prod(x.shape[0:])
print(type(input_shape))