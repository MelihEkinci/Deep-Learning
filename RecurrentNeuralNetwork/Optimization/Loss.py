import numpy as np
class CrossEntropyLoss:
    def __init__(self):
        #BaseLayer().__init__()
        #self.trainable=False
        #self.input_tensor=None
        self.prediction_tensor=None
        self.old_x=None
    
    def forward(self, prediction_tensor, label_tensor):
        
        self.old_x = prediction_tensor.clip(min=np.finfo(float).eps,max=None)
        #self.old_x=prediction_tensor
        self.old_y = prediction_tensor
        return (np.where(label_tensor==1,-np.log(self.old_x), 0)).sum()

    def backward(self, label_tensor):
        #anss=-label_tensor/(self.prediction_tensor + 10**-100)# -y_true/(y_pred + 10**-100)
        #return np.where(label_tensor==1,-1/self.old_x, 0)
        #return anss
        return - (label_tensor / self.old_x)