from Layers.Base import BaseLayer
#from Base import BaseLayer
import numpy as np

class Conv(BaseLayer):

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.is1D = False if len(convolution_shape) >= 3 else True
        self.is1x1 = True if convolution_shape[1:] == (1, 1) else False
        #print(self.is1D)
        self.num_kernels = num_kernels
        self.filters = np.random.uniform(0, 1, size=(num_kernels, *convolution_shape))
        self.weights = self.filters
        self.bias = np.random.uniform(0, 1, size=(num_kernels, 1))
        self.input_tensor=None
        self._optimizer = None

    @property
    def optimizer(self):

        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer_value):

        self._optimizer = optimizer_value
    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    def pad_zeros(self, x, pad_before,pad_after):
        #print('In zero pad',x.shape)
        #print((pad_before[1], pad_after[1]))
        if not self.is1D:
            return np.pad(x, ((0, 0), (0, 0),(pad_before[0], pad_after[0]), (pad_before[1],pad_after[1])), 'constant', constant_values=0)
        else:
            return np.pad(x, ((0, 0), (0, 0), (pad_before[0], pad_after[0])), 'constant',constant_values=0)

    def convolve(self,X, W, b):
        """
        X is the input, and W is the filter,b is the bias
        """
        assert X.shape == W.shape
        Z = np.sum(np.multiply(W, X)) + b
        A = np.maximum(0,Z) #Relu
        return A

    def forward(self, input_tensor):
        self.filters = self.weights

        kernel_sizes=np.array(self.convolution_shape[1:])
        pad_before=np.floor(kernel_sizes/2).astype(int)
        pad_after=kernel_sizes-pad_before-1

        m, n_C_prev, n_Y_prev = input_tensor.shape[0:3]
        strideY = self.stride_shape[0]
        n_C, _, fy = self.filters.shape[0:3]
        n_Y = (n_Y_prev + pad_before[0] + pad_after[0] - fy) // strideY + 1

        if self.is1D:
            output = np.zeros((m, n_C, n_Y))
        else:
            n_X_prev = input_tensor.shape[-1]
            fx = self.filters.shape[-1]
            strideX=self.stride_shape[1]
            n_X = (n_X_prev + pad_before[1]+pad_after[1] - fx) // strideX + 1
            output = np.zeros((m, n_C, n_Y, n_X))


        #print("output_shape",output.shape)
        padded_input = self.pad_zeros(input_tensor,pad_before,pad_after)
        #print("filter shape",self.filters.shape)
        for i in range(m):
            # looping over batches
            padded_img = padded_input[i]
            for c in range(n_C):
                # corresponding filter and bias for channel
                fil = self.filters[c]
                b = self.bias[c]

                if not self.is1D:
                    for x in range(n_X):
                        for y in range(n_Y):
                            x_range = (strideX * x, strideX * x + fx)
                            y_range = (strideY * y, strideY * y + fy)
                            input_slice = padded_img[:,y_range[0]:y_range[1], x_range[0]:x_range[1]]
                            output[i, c, y, x] = self.convolve(input_slice, fil, b)
                else:
                    for y in range(n_Y):
                        y_range = (strideY * y, strideY * y + fy)
                        input_slice = padded_img[:,y_range[0]:y_range[1]]
                        output[i, c, y] = self.convolve(input_slice, fil, b)

        #print(1)
        self.input_tensor=input_tensor
        return output

    def backward(self,error_tensor):
        kernel_sizes = np.array(self.convolution_shape[1:])
        pad_before = np.floor(kernel_sizes / 2).astype(int)
        pad_after = kernel_sizes - pad_before - 1

        if self.is1D:
            #retrieving shapes from input, weights and error tensor
            (m, n_C_prev, n_Y_prev) = self.input_tensor.shape
            (n_C, n_C_prev, fy) = self.weights.shape
            strideY = self.stride_shape[0]
            (m, n_C, n_Y) = error_tensor.shape
            #initializing as zeros
            der_input = np.zeros((m, n_C_prev,n_Y_prev ))
            der_weights = np.zeros((n_C,n_C_prev,fy))
            der_bias = np.zeros((n_C, 1, 1))

        else:
            #retrieving shapes from input, weights and error tensor
            (m,n_C_prev, n_Y_prev, n_X_prev) = self.input_tensor.shape
            (n_C,n_C_prev,fy,fx) = self.weights.shape
            strideY, strideX=self.stride_shape
            (m, n_C,n_Y, n_X) = error_tensor.shape
            # initializing as zeros
            der_input = np.zeros((m, n_C_prev,n_Y_prev, n_X_prev ))
            der_weights = np.zeros((n_C,n_C_prev,fy,fx))
            der_bias = np.zeros((n_C, 1, 1, 1))

        # padding
        input_pat = self.pad_zeros(self.input_tensor, pad_before,pad_after)
        der_input_pad = self.pad_zeros(der_input, pad_before,pad_after)

        for i in range(m):  #convolution loop
            # select the batches for derivative and input
            input_pat1 = input_pat[i]
            der_input_pad2 = der_input_pad[i]

            for y in range(n_Y):  #  loop for Y
                if not self.is1D:
                    for x in range(n_X):  # loop for X
                        for c in range(n_C):  # loop for channels
                            # find the borders of slice
                            y_start = y*strideY
                            y_end = y_start + fy
                            x_start = x*strideX
                            x_end = x_start + fx

                            # use the borders to find the slice
                            input_slice = input_pat1[:,y_start:y_end, x_start:x_end]

                            # calculating gradients and error tensor for next layer
                            der_input_pad2[:,y_start:y_end, x_start:x_end] += self.weights[c,:,:, : ] * error_tensor[i, c, y, x]
                            der_weights[c,:, :, :] += input_slice * error_tensor[i, c,y, x]
                            der_bias[c,:, :, :] += error_tensor[i, c, y, x]
                else:
                    for c in range(n_C):   # loop for channels
                        # find the borders of slice
                        y_start = y*strideY
                        y_end = y_start + fy

                        # use the borders to find the slice
                        input_slice = input_pat1[:, y_start:y_end]

                        # calculating gradients and error tensor for next layer
                        der_input_pad2[:, y_start:y_end] += self.weights[c, :, :] * error_tensor[i, c, y]
                        der_weights[c, :, :] += input_slice * error_tensor[i, c, y]
                        der_bias[c, :, :] += error_tensor[i, c, y]
            if not self.is1x1: ##not 1x1 convolution
                if not self.is1D: ##1d dimensionality
                    # retrieve non-pad image
                    der_input[i, :, :, :] = der_input_pad2[:,pad_before[0]:-pad_after[0], pad_before[1]:-pad_after[1]]
                else:
                    der_input[i, :, :] = der_input_pad2[:, pad_before[0]:-pad_after[0]]
            else: ##1x1 conv
                if not self.is1D: ##1d dimensionality
                    # retrieve unpaded for 1x1
                    der_input[i, :, :, :] = der_input_pad2[:,:,:]
                else:
                    der_input[i, :, :] = der_input_pad2[:,:]

        ### END CODE HERE ###

        self.gradient_weights=der_weights
        self.gradient_bias=der_bias

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return der_input

    def initialize(self,weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.num_kernels, *self.convolution_shape), np.prod(self.convolution_shape),np.prod(self.convolution_shape[1:]) * self.num_kernels)
        self.bias = bias_initializer.initialize((self.num_kernels, 1), self.num_kernels,1)



'''
batch_size = 2
input_shape = (3, 10, 14)
input_size = 14 * 10 * 3
uneven_input_shape = (3, 11, 15)
uneven_input_size = 15 * 11 * 3
spatial_input_shape = np.prod(input_shape[1:])
kernel_shape = (3, 5, 8)
num_kernels = 4
hidden_channels = 3



#def test_backward_size_stride(self):
conv = Conv((3, 2), kernel_shape, num_kernels)
input_tensor = np.array(range(np.prod(input_shape) * batch_size), dtype=float)
input_tensor = input_tensor.reshape(batch_size, *input_shape)
output_tensor = conv.forward(input_tensor)
error_tensor = conv.backward(output_tensor)
assert(error_tensor.shape == (batch_size, *input_shape))
print("assert try 2")


#def test_1x1_convolution(self):
conv = Conv((1, 1), (3, 1, 1), num_kernels)
input_tensor = np.array(range(input_size * batch_size), dtype=float)
input_tensor = input_tensor.reshape(batch_size, *input_shape)
output_tensor = conv.forward(input_tensor)
print(output_tensor.shape,(batch_size, num_kernels, *input_shape[1:]))
assert(output_tensor.shape==(batch_size, num_kernels, *input_shape[1:]))

error_tensor = conv.backward(output_tensor)
print(error_tensor.shape,(batch_size, *input_shape))
assert(error_tensor.shape==(batch_size, *input_shape))
'''

'''''''''

#test forward
np.random.seed(1337)
conv = Conv((1, 1), (1, 3, 3), 1)
conv.weights = (1./15.) * np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]])
conv.bias = np.array([0])
conv.weights = np.expand_dims(conv.weights, 0)
input_tensor = np.random.random((1, 1, 10, 14))
expected_output = gaussian_filter(input_tensor[0, 0, :, :], 0.85, mode='constant', cval=0.0, truncate=1.0)
output_tensor = conv.forward(input_tensor).reshape((10, 14))
difference = np.max(np.abs(expected_output - output_tensor))

print(expected_output)
print('________________________')
print(output_tensor)
#print(expected_output)


#def test_backward_size(self):
conv = Conv((1, 1), kernel_shape, num_kernels)
input_tensor = np.array(range(np.prod(input_shape) * batch_size), dtype=float)
input_tensor = input_tensor.reshape(batch_size, *input_shape)
output_tensor = conv.forward(input_tensor)
error_tensor = conv.backward(output_tensor)
assert(error_tensor.shape ==(batch_size, *input_shape))
print("assert try 1")


#def test_backward_size_stride(self):
conv = Conv((3, 2), kernel_shape, num_kernels)
input_tensor = np.array(range(np.prod(input_shape) * batch_size), dtype=float)
input_tensor = input_tensor.reshape(batch_size, *input_shape)
output_tensor = conv.forward(input_tensor)
error_tensor = conv.backward(output_tensor)
assert(error_tensor.shape == (batch_size, *input_shape))
print("assert try 2")


#test_1D_backward_size(self):
conv = Conv([2], (3, 3), num_kernels)
input_tensor = np.array(range(45 * batch_size), dtype=float)
input_tensor = input_tensor.reshape((batch_size, 3, 15))
output_tensor = conv.forward(input_tensor)
error_tensor = conv.backward(output_tensor)
assert(error_tensor.shape==(batch_size, 3, 15))

print("assert try 3")

#test stride shape

conv = Conv((3, 2), kernel_shape, num_kernels)
input_tensor = np.array(range(int(np.prod(input_shape) * batch_size)), dtype=float)
input_tensor = input_tensor.reshape(batch_size, *input_shape)
output_tensor = conv.forward(input_tensor)
assert(output_tensor.shape == (batch_size, num_kernels, 4, 7))
'''
#test forward size
'''
conv = Conv((1, 1), kernel_shape, num_kernels)
input_tensor = np.array(range(int(np.prod(input_shape) * batch_size)), dtype=float)
input_tensor = input_tensor.reshape(batch_size, *input_shape)
print("input tensor shape",input_tensor.shape)
output_tensor = conv.forward(input_tensor)
print("output_shapes",output_tensor.shape,(batch_size, num_kernels, *input_shape[1:]))
assert(output_tensor.shape== (batch_size, num_kernels, *input_shape[1:]))
'''

