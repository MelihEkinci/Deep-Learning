from Layers.Base import BaseLayer
from Optimization.Optimizers import Sgd
import numpy as np


class FullyConnected(BaseLayer, Sgd):

    def __init__(self, input_size, output_size):
        BaseLayerobj = BaseLayer()
        self.trainable = True
        self._optimizer = None
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = None
        # self.bias= 0.1

        self.weights = np.random.rand(self.input_size + 1, self.output_size)

        self.input_tensor = None
        self.gradient_weights = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    def forward(self, input_tensor):
        self.input_tensor = np.concatenate((input_tensor, np.ones((input_tensor.shape[0], 1))), axis=1)

        self.forward_output = np.dot(self.input_tensor, self.weights)
        return self.forward_output

    def backward(self, error_tensor):
        self.backward_output = np.dot(error_tensor, self.weights.transpose())
        self.gradient_weights = np.dot(self.input_tensor.transpose(), error_tensor)
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return self.backward_output[:, :-1]  # previous layer error_tensor