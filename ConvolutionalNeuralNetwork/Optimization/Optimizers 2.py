import numpy as np


class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        # calculates gradient
        step_size = self.learning_rate * gradient_tensor
        updated_weight_tensor = weight_tensor - step_size

        return updated_weight_tensor


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.change = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        # calculates gradient
        step_size = self.learning_rate * gradient_tensor
        self.change = self.momentum * self.change - step_size  # updated change
        # print('Original::',weight_tensor)
        updated_weight_tensor = weight_tensor + self.change
        # print('Updated::',updated_weight_tensor)
        return updated_weight_tensor


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.change1 = 0
        self.change2 = 0
        self.k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        # calculates gradient
        self.k += 1

        g = gradient_tensor
        self.change1 = self.mu * self.change1 + (1 - self.mu) * g
        xx=np.multiply((1 - self.rho) * g, g)
        #xx = np.dot(g, g)*(1 - self.rho)
        self.change2 = self.rho * self.change2 + xx

        bias_corrected_v = self.change1 / (1 - (self.mu ** self.k))
        bias_corrected_r = self.change2 / (1 - (self.rho ** self.k))

        # print('Original::',weight_tensor)
        updated_weight_tensor = weight_tensor - self.learning_rate * (
                    bias_corrected_v / (np.sqrt(bias_corrected_r) + np.finfo(float).eps))
        # print('Updated::',updated_weight_tensor)
        return updated_weight_tensor
