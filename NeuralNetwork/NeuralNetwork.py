class NeuralNetwork():
    def __init__(self, optimizer):

        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

        self.input_tensor_new = None
        self.label_tensor_new = None
        self.test_data = np.array([])

    def forward(self):
        # self.input_tensor = np.random.rand(self.batch_size, self.input_size) weight= input size * output size 4*3
        if self.test_data.size != 0:
            print('Testing Lol')
            self.input_tensor_new = self.test_data
        else:
            self.input_tensor_new, self.label_tensor_new = self.data_layer.next()
        # print(self.input_tensor_new.shape)

        for layer in self.layers[:]:
            ans = layer.forward(self.input_tensor_new)
            self.input_tensor_new = ans  # probability at end by softmax

        # print('shape of ans',ans.shape)

        anss = self.loss_layer.forward(ans, self.label_tensor_new)
        # print(anss,'singular in ')
        return anss
        # super().__init__(self.input_tensor_new.shape[0],self.input_tensor_new.shape[1])

    #         print(self.input_tensor_new.shape)
    #         print(self.weights.shape)
    # return super().forward(self.input_tensor_new)

    def backward(self):

        loss_cross = self.loss_layer.backward(self.label_tensor_new)

        for layer in self.layers[:]:
            ans = layer.backward(loss_cross)

        # print(loss_cross)
        return ans

    def append_layer(self, layer):
        if layer.trainable:
            layer._optimizer = copy.deepcopy(self.optimizer)  # layer._optimizer
        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            curr_loss = self.forward()
            self.loss.append(curr_loss)
            self.backward()

        return curr_loss

    def test(self, input_tensor):
        self.test_data = input_tensor
        self.forward()
        return self.input_tensor_new

