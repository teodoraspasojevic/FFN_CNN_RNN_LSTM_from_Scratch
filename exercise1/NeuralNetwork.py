import copy


class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        self.label_tensor = label_tensor
        for i in range(len(self.layers)):
            input_tensor = self.layers[i].forward(input_tensor)
        output_tensor = input_tensor
        # predicted_class = np.argmax(output_tensor)
        loss = self.loss_layer.forward(output_tensor, self.label_tensor)
        return loss
        # TODO: check if this should be the output

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for i in range(len(self.layers)-1, -1, -1):
            error_tensor = self.layers[i].backward(error_tensor)
        return error_tensor

    def append_layer(self, layer):
        if layer.trainable:
            optimizer_copy = copy.deepcopy(self.optimizer)
            layer.optimizer = optimizer_copy
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        for i in range(len(self.layers)):
            input_tensor = self.layers[i].forward(input_tensor)
        output_tensor = input_tensor
        return output_tensor
