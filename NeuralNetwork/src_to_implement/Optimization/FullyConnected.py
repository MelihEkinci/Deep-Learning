import numpy as np
import matplotlib.pyplot as plt

class FullyConnected(BaseLayer, Sgd):

    def __init__(self, input_size, output_size):
        BaseLayerobj = BaseLayer()
        BaseLayerobj.trainable = True
        self._optimizer = None
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.zeros((output_size, input_size))
        self.weights[:] = np.linspace(0, 1, num=input_size)  # uniformaly distributed between 0 to 1
        self.allTensors = {}
        self.index = 0
        self.input_tensor = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    def forward(self, input_tensor):
        ans = input_tensor.dot(np.transpose(self.weights))
        self.allTensors[self.index] = input_tensor
        self.index += 1
        print('in forward', ans.shape)
        self.input_tensor = input_tensor  # test
        return ans  # for next layer

    def backward(self, error_tensor):
        error_1 = error_tensor.dot((self.weights))  # weight * error_tensor

        if self._optimizer != None:
            sgd_obj = Sgd(1)
            # print('Before',self.weights)
            self.weights = sgd_obj.calculate_update(self.weights, np.dot(self.input_tensor.T, error_tensor))
            # print('After',self.weights)
        return error_1
