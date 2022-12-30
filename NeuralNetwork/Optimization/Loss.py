import numpy as np
from Layers.Base import BaseLayer
from Optimizers import Sgd


class CrossEntropyLoss:
    def __init__(self):
        # BaseLayer().__init__()
        # self.trainable=False
        # self.input_tensor=None
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        # self.input_tensor=input_tensor
        # return np.maximum(0,input_tensor)

        self.prediction_tensor = prediction_tensor
        self.old_x = prediction_tensor.clip(min=np.finfo(float).eps, max=None)
        # self.old_x=prediction_tensor
        self.old_y = prediction_tensor
        return (np.where(label_tensor == 1, -np.log(self.old_x), 0)).sum()

        '''        
        sum_score = 0.0
        for i in range(len(label_tensor)):
            for j in range(len(label_tensor[i])):
                sum_score += label_tensor[i][j] * np.log(np.finfo(float).eps + self.prediction_tensor[i][j])

        return sum_score


        print(label_tensor)
        print(prediction_tensor)
        m=label_tensor.shape[0]

        log_likelihood =-np.log(prediction_tensor[range(m),label_tensor]+np.finfo(np.float64).min)
        loss = np.sum(log_likelihood) / m
        return loss
        '''

    def backward(self, label_tensor):
        # bb=-label_tensor/(self.prediction_tensor+np.finfo(float).eps)
        # print(bb)
        return np.where(label_tensor == 1, -1 / self.old_x, 0)