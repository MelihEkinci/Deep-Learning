import numpy as np


class L1_Regularizer:

    def __init__(self, alpha):
        self.alpha = alpha

    def norm(self, weight):
        return np.sum(np.abs(weight)) * self.alpha
         #None

    def calculate_gradient(self, weights):
        return self.alpha * np.sign(weights)


class L2_Regularizer:

    def __init__(self, alpha):
        self.alpha = alpha

    def norm(self, weight):
        return np.sum(np.square(weight)) * self.alpha
        #return None

    def calculate_gradient(self, weights):
        return weights * self.alpha
