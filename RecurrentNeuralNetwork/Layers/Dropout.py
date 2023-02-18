import numpy as np
from Layers.Base import BaseLayer


class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.p = probability
        self.binary_value = None

    def forward(self, input_tensor):
        if self.testing_phase == True:
            return input_tensor
        self.binary_value = np.random.rand(input_tensor.shape[0], input_tensor.shape[1]) < self.p

        res = np.multiply(input_tensor, self.binary_value)
        res /= self.p
        return res

    def backward(self, error_tensor):
        res = np.multiply(error_tensor, self.binary_value)
        res /= self.p
        return res
