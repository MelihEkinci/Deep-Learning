from Layers.Base import BaseLayer
# from Base import BaseLayer
import numpy as np


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.input_shape = channels
        self.trainable = True

        # Batch mean & var must be defined during training
        self.mu = np.zeros(channels)
        self.var = np.ones(channels)

        # Trainable parameters
        self.beta = None
        self.gamma = None
        self.initialize(channels)  # initialises the beta and gamma

        self.epsilon = 1e-11
        # Exponential moving average for mu & var update
        self.k = 0
        self.momentum = 0.8

        # self.inp_shape=None
        self.input_tensor = None
        self.first = 0
        self.xHat = None

        self.weights = None
        self.bias = None
        self._gradient_weights = None
        self._gradient_bias = None
        self.xmu = None

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


    def initialize(self, shape):
        self.beta = np.zeros(shape)
        self.gamma = np.ones(shape)

    def forward(self, input_tensor):
        self.k += 1
        self.batch_size = input_tensor.shape[0]
        self.input_tensor = input_tensor
        isConv = len(input_tensor.shape) == 4

        if isConv:
            input_tensor = self.reformat(input_tensor)

        if self.testing_phase == False:

            if (self.batch_size == 0):
                # First iteration : save batch_size
                self.batch_size = input_tensor.shape[0]

            # Training : compute BN pass
            batch_mu = np.mean(input_tensor, axis=0)
            batch_var = np.var(input_tensor, axis=0)

            x_normalized = (input_tensor - batch_mu) / np.sqrt(batch_var + self.epsilon)

            #print(self.gamma.shape, x_normalized.shape, self.beta.shape)
            x_bn = self.gamma * x_normalized + self.beta
            #print('XBn shape', x_bn.shape)

            # Update mu & var
            if self.first == 0:
                self.mu = batch_mu
                self.var = batch_var
                self.first += 1
            self.mu = self.momentum * self.mu + (1-self.momentum) * batch_mu
            self.var = self.momentum * self.var + (1 - self.momentum) * batch_var

        else:  # testing

            estimated_mu = self.mu
            estimated_var = self.var

            x_normalized = (input_tensor - estimated_mu) / np.sqrt(estimated_var + self.epsilon)
            x_bn = self.gamma * x_normalized + self.beta

        if isConv:
            x_bn = self.reformat(x_bn)

        self.weights = self.gamma
        self.bias = self.beta


        self.xmu = input_tensor - self.mu
        self.xHat = x_bn#self.xmu * (1/np.sqrt(self.var + self.epsilon)) # for backward
        return x_bn

    def backward(self,error_tensor):
        # unfold the variables stored in cache

        # get the dimensions of the input/output
        if error_tensor.shape == 4:
            error_tensor = self.reformat(error_tensor) #for conv

        N, D = error_tensor.shape

        # step9
        dbeta = np.sum(error_tensor, axis=0)
        print('sssac', dbeta)

        dgammax = error_tensor  # not necessary, but more understandable

        # step8
        dgamma = np.sum(dgammax * self.xHat, axis=0)
        print('dddd',dgamma)
        dxhat = dgammax * self.gamma

        sqrtvar = np.sqrt(self.var + self.epsilon)
        # step7
        divar = np.sum(dxhat * self.xmu, axis=0)
        dxmu1 = dxhat * (1/sqrtvar)

        # step6
        dsqrtvar = -1. / (sqrtvar ** 2) * divar

        # step5
        dvar = 0.5 * 1. / np.sqrt(self.var + self.epsilon) * dsqrtvar

        # step4
        dsq = 1. / N * np.ones((N, D)) * dvar

        # step3
        dxmu2 = 2 * self.xmu * dsq

        # step2
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)

        # step1
        dx2 = 1. / N * np.ones((N, D)) * dmu

        # step0
        dx = dx1 + dx2

        if error_tensor.shape == 4:
            error_tensor = self.reformat(error_tensor) #for conv

        self._gradient_weights = dgamma
        self._gradient_bias = dbeta
        return dx#, dgamma, dbeta


    def reformat(self, tensor):
        if len(tensor.shape) == 4:
            return np.concatenate(tensor.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2] * tensor.shape[3]),
                                  axis=1).T
        elif len(tensor.shape) == 2:
            return np.transpose(
                tensor.reshape(self.input_tensor.shape[0], self.input_tensor.shape[2] * self.input_tensor.shape[3],
                               self.input_tensor.shape[1]), (0, 2, 1)).reshape(self.input_tensor.shape[0],
                                                                               self.input_tensor.shape[1],
                                                                               self.input_tensor.shape[2],
                                                                               self.input_tensor.shape[3])
        else:
            return tensor


