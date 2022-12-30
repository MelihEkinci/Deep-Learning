from Layers.Base import BaseLayer
import numpy as np
#from Layers import *
#from Base import BaseLayer
class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):

        super().__init__()
        self.backward_output = None
        self.input_tensor = None
        self._optimizer = None
        self._gradient_weights = None
        self.forward_output = None
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        # Initialize weights to carry bias in last row
        self.weights = np.random.rand(self.input_size + 1, self.output_size)
    
    def initialize(self,weights_initializer, bias_initializer):
        #self.weights[:-1] = weights_initializer.initialize(
        #print('Lol in fully connected')
        self.weights[:-1]= weights_initializer.initialize((self.input_size,self.output_size),self.input_size,self.output_size)
        self.weights[-1:] = bias_initializer.initialize((self.input_size,self.output_size),1,self.output_size)
        #print('Hellooo',self.weights)
        
    
    def forward(self, input_tensor):
        # wx+b*1

        self.input_tensor = np.concatenate((input_tensor, np.ones((input_tensor.shape[0], 1))), axis=1)

        self.forward_output = np.dot(self.input_tensor, self.weights)
        return self.forward_output

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

    def backward(self, error_tensor):

        self.gradient_weights = np.dot(self.input_tensor.transpose(), error_tensor)
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        self.backward_output = np.dot(error_tensor, self.weights.transpose())

        return self.backward_output[:, :-1]
