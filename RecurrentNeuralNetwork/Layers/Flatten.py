from Layers.Base import BaseLayer
import numpy as np
class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.input_tensor=None

    def forward(self, input_tensor):
        self.input_tensor=input_tensor
        input_shape= np.prod(self.input_tensor.shape[1:])#self.input_tensor.shape[1]*self.input_tensor.shape[2]*self.input_tensor.shape[3]
        return input_tensor.reshape(self.input_tensor.shape[0],input_shape)

    def backward(self, error_tensor):
        return error_tensor.reshape(*self.input_tensor.shape)
    