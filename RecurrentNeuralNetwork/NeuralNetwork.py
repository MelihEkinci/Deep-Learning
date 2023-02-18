# Neural Network 1
import copy
from Layers.Base import BaseLayer


class NeuralNetwork(BaseLayer):
    # __init__ is the constructor class
    def __init__(self, optimizer_value, weight_initializer, bias_initializer):
        super().__init__()
        self.value_out = None
        self.value_in = None
        self.optimizer = optimizer_value
        self.loss = list()
        self.data_layer = None
        self.layers = list()
        self.loss_layer = None
        self.initializers = weight_initializer, bias_initializer

    def forward(self):
        self.value_in, self.value_out = self.data_layer.next()
        for layer in self.layers:
            self.value_in = layer.forward(self.value_in)

        self.value_in = self.loss_layer.forward(self.value_in, self.value_out)

        if self.optimizer.regularizer is not None:
            regularization_loss = self.optimizer.regularizer.norm(layer.weights)
            self.value_in += regularization_loss

        self.loss.append(self.value_in)
        self.value_out = self.loss_layer.backward(self.value_out)
        return self.value_in

    def backward(self):
        # value_out = self.value_out
        for layer in reversed(self.layers):
            self.value_out = layer.backward(self.value_out)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(*self.initializers)

        self.layers.append(layer)

    def train(self, iterations):
        self.phase = False
        for i in range(iterations):

            self.forward()
            self.backward()

    def test(self, input_tensor):

        self.phase = True
        value_in = input_tensor
        for layer in self.layers:
            value_in = layer.forward(value_in)

        return value_in

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase_value):
        self._phase = phase_value
        self.testing_phase = phase_value
