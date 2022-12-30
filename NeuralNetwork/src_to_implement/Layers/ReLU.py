import numpy as np
from Layers.Base import BaseLayer
from Optimizers import Sgd

class ReLU(BaseLayer):
    def __init__(self):
        # BaseLayer().__init__()
        self.trainable = False
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        # error_tensor = 10*5 input_tensor=10*5

        #         print('np.maximum(0,self.input_tensor)=',np.maximum(0,self.input_tensor))
        #         print('error_tensor= ',error_tensor)
        anss = np.maximum(0, self.input_tensor)
        # print('hbhb',anss)
        anss = (anss > 0) * 1
        # print('hbhaaa',anss)
        return anss * error_tensor  # np.maximum(0,self.input_tensor)*error_tensor
        # return np.maximum(0,error_tensor)