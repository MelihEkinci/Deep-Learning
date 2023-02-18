import numpy as np
from Layers.Base import BaseLayer
#from Base import BaseLayer

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape=stride_shape
        self.pooling_shape = pooling_shape
        self.data=None
        super().__init__()
        self.trainable = False

    def forward(self,input_tensor):
        # setting shapes
        (m,  n_C_prev, n_Y_prev, n_X_prev) = input_tensor.shape
        sY, sX = self.stride_shape
        pY, pX = self.pooling_shape

        # initializing the dimensions
        n_Y = int(1 + (n_Y_prev - pY) / sY)
        n_X = int(1 + (n_X_prev - pX) / sX)
        n_C = n_C_prev

        # Initialziing as zeros
        A = np.zeros((m, n_C, n_Y, n_X))


        for i in range(m):
            for y in range(n_Y):
                for x in range(n_X):
                    for c in range(n_C):

                        # borders for the slice
                        y_start = y * sY
                        y_end = y_start + pY
                        x_start = x * sX
                        x_end = x_start + pX

                        # getting the slice
                        input_slice = input_tensor[i,c, y_start:y_end, x_start:x_end]

                        # compute max pooling.

                        A[i,c, y, x] = np.max(input_slice)

        self.data=input_tensor

        return A


    def backward(self,error_tensor):

        # setting the dimensions
        sY, sX = self.stride_shape
        pY, pX = self.pooling_shape
        m,n_C, n_Y, n_X = error_tensor.shape

        # Initializing with zeros
        der_input = np.zeros(self.data.shape)

        for i in range(m):
            input1 = self.data[i]
            for y in range(n_Y):
                for x in range(n_X):
                    for c in range(n_C):
                        # Find the borders
                        y_start = y * sY
                        y_end = y_start + pY
                        x_start = x * sX
                        x_end = x_start + pX

                        input_slice = input1[c,y_start:y_end, x_start:x_end]
                        # max pooling for mask
                        mask = input_slice == np.max(input_slice)
                        # calculating the derivative
                        der_input[i,c, y_start:y_end, x_start:x_end] += np.multiply(mask,error_tensor[i,c, y, x])
        return der_input


'''
batch_size = 2
input_shape = (2, 4, 7)
input_size = np.prod(input_shape)

np.random.seed(1337)
input_shape = (2, 2, 2)
input_tensor = np.random.uniform(-1, 1, (batch_size, *input_shape))
print(input_tensor)
#input_tensor=np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], [[17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]]])
categories = 12
label_tensor = np.zeros([batch_size, categories])
for i in range(batch_size):
    label_tensor[i, np.random.randint(0, categories)] = 1

#test shape
layer = Pooling((2, 2), (2, 2))
result = layer.forward(input_tensor)
expected_shape = np.array([batch_size, 2, 2, 3])

print(result)
assert(np.sum(np.abs(np.array(result.shape) - expected_shape))== 0)
'''