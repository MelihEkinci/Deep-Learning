class BaseLayer:
    def __init__(self):
        self.trainable=False
        self.testing_phase = False
        #self.weight=np.array([0.1])