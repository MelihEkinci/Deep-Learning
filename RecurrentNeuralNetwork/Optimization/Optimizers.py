import numpy as np
#from Optimization.Constraints import *
#import Constraints

class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, value):
        self.regularizer = value
        return self.regularizer


class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        # calculates gradient
        if self.regularizer is not None:
            #print('Hi')
            gradient_tensor = self.regularizer.calculate_gradient(weight_tensor) + gradient_tensor

        step_size = self.learning_rate * gradient_tensor
        updated_weight_tensor = weight_tensor - step_size

            #updated_weight_tensor = ((1 - self.learning_rate * self.regularizer.alpha) * weight_tensor) - step_size
            #updated_weight_tensor = weight_tensor - (self.learning_rate * self.regularizer.alpha) * np.sign(weight_tensor) - step_size
        return updated_weight_tensor


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.change = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        # calculates gradient

        #if self.regularizer is not None:
        #    gradient_tensor = self.regularizer.calculate_gradient(weight_tensor) + gradient_tensor

        step_size = self.learning_rate * gradient_tensor
        self.change = self.momentum * self.change - step_size  # updated change

        if self.regularizer is not None:
            updated_weight_tensor = weight_tensor + self.change -  self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        else:
            updated_weight_tensor = weight_tensor + self.change

        return updated_weight_tensor


class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.change1 = 0
        self.change2 = 0
        self.k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        # calculates gradient
        self.k += 1

        #if self.regularizer is not None:
        # gradient_tensor = self.regularizer.calculate_gradient(weight_tensor) + gradient_tensor

        g = gradient_tensor
        self.change1 = self.mu * self.change1 + (1 - self.mu) * g


        xx = np.multiply((1 - self.rho) * g, g)
        # xx = np.dot(g, g)*(1 - self.rho)
        self.change2 = self.rho * self.change2 + xx


        bias_corrected_v = self.change1 / (1 - (self.mu ** self.k))
        bias_corrected_r = self.change2 / (1 - (self.rho ** self.k))

        # print('Original::',weight_tensor)
        updated_weight_tensor = weight_tensor - self.learning_rate * (
                (bias_corrected_v / (np.sqrt(bias_corrected_r) + np.finfo(float).eps)) + self.regularizer.calculate_gradient(weight_tensor))
        # print('Updated::',updated_weight_tensor)
        return updated_weight_tensor


# setup for tests
delta = 0.1
regularizer_strength = 1337
shape = (4, 5)

'''
#test l1
regularizer = Constraints.L1_Regularizer(regularizer_strength)

weights_tensor = np.ones(shape)
weights_tensor[1:3, 2:4] *= -2
weights_tensor = regularizer.calculate_gradient(weights_tensor)

expected = np.ones(shape) * regularizer_strength
expected[1:3, 2:4] *= -1

print('expected',expected)
print('actual',weights_tensor)

difference = np.sum(np.abs(weights_tensor - expected))
assert(difference <= 1e-10)
print(difference)
'''
'''
#test l1 norm
regularizer = Constraints.L1_Regularizer(regularizer_strength)
weights_tensor = np.ones(shape)
weights_tensor[1:3, 2:4] *= -2
norm = regularizer.norm(weights_tensor)
np.testing.assert_almost_equal(norm, 24*regularizer_strength)
'''
'''
#test l2
regularizer = Constraints.L2_Regularizer(regularizer_strength)

weights_tensor = np.ones(shape)
weights_tensor = regularizer.calculate_gradient(weights_tensor)

difference = np.sum(np.abs(weights_tensor - np.ones(shape) * regularizer_strength))
assert(difference<= 1e-10)
'''
'''
#test l2 norm

regularizer = Constraints.L2_Regularizer(regularizer_strength)

weights_tensor = np.ones(shape)
weights_tensor[1:3, 2:4] += 1
norm = regularizer.norm(weights_tensor)
np.testing.assert_almost_equal(norm, 32 * regularizer_strength)
'''
'''
# test l1reg with sgd

weights_tensor = np.ones(shape)
weights_tensor[1:3, 2:4] *= -1

optimizer = Sgd(2)
regularizer = Constraints.L1_Regularizer(2)
optimizer.add_regularizer(regularizer)

result = optimizer.calculate_update(weights_tensor, np.ones(shape) * 2)
print(result)
result = optimizer.calculate_update(result, np.ones(shape) * 2)
print('again',result)
np.testing.assert_almost_equal(np.sum(result), -116, 2)
'''

'''
#test l2 with sgd
weights_tensor = np.ones(shape)
weights_tensor[1:3, 2:4] *= -1

optimizer = Sgd(2)
regularizer = Constraints.L2_Regularizer(2)
optimizer.add_regularizer(regularizer)

result = optimizer.calculate_update(weights_tensor, np.ones(shape)*2)
result = optimizer.calculate_update(result, np.ones(shape) * 2)

np.testing.assert_almost_equal(np.sum(result), 268, 2)
'''

'''
#Sgd with momentum l1loss
weights_tensor = np.ones(shape)
weights_tensor[1:3, 2:4] *= -1

optimizer = SgdWithMomentum(2,0.9)
regularizer = Constraints.L1_Regularizer(2)
optimizer.add_regularizer(regularizer)

result = optimizer.calculate_update(weights_tensor, np.ones(shape)*2)
result = optimizer.calculate_update(result, np.ones(shape) * 2)

np.testing.assert_almost_equal(np.sum(result), -188, 1)
'''
'''
weights_tensor = np.ones(shape)
weights_tensor[1:3, 2:4] *= -1

optimizer = Adam(2, 0.9, 0.999)
regularizer = Constraints.L1_Regularizer(2)
optimizer.add_regularizer(regularizer)

result = optimizer.calculate_update(weights_tensor, np.ones(shape)*2)
result = optimizer.calculate_update(result, np.ones(shape) * 2)

np.testing.assert_almost_equal(np.sum(result), -68, 2)
'''