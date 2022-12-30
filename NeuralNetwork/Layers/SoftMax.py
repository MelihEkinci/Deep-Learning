import numpy as np
from Layers.Base import BaseLayer
from Optimizers import Sgd

class SoftMax(BaseLayer):
    def __init__(self):
        # BaseLayer().__init__()
        self.trainable = False
        self.input_tensor = None

    def forward(self, input_tensor):
        #         denominator=np.exp(input_tensor-np.max(input_tensor))
        #         print('s',denominator)
        #         return sum(denominator/np.sum(denominator))
        self.input_tensor = input_tensor
        s = np.max(input_tensor, axis=1)
        s = s[:, np.newaxis]
        e_x = np.exp(input_tensor - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]
        self.res = e_x / div
        return self.res

    def backward(self, error_tensor):
        # aa=np.dot(self.input_tensor,(error_tensor-np.sum(error_tensor,self.input_tensor)))
        bb = self.res * (error_tensor - (error_tensor * self.res).sum(axis=1)[:, None])
        return bb