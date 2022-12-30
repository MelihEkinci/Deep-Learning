import numpy as np
import matplotlib.pyplot as plt


class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        # calculates gradient
        step_size = self.learning_rate * gradient_tensor
        updated_weight_tensor = weight_tensor - step_size

        return updated_weight_tensor
