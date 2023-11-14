import copy


class NeuralNetwork:
    def __init__(self, optimizer, loss: list, layers: list, data_layer, loss_layer):
        self.optimizer = optimizer
        self.loss = loss
        self.layers = layers
        self.data_layer = data_layer
        self.loss_layer = loss_layer
        self.label_tensor = None

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        self.label_tensor = label_tensor
        for i in range(len(self.layers) - 1):
            input_tensor = self.layers[i].forward(input_tensor)
        loss = self.layers[-1].forward(input_tensor, label_tensor)
        return loss

    def backward(self):
        error_tensor = self.layers[-1].backward(self.label_tensor)
        for i in range(len(self.layers)-2, -1):
            error_tensor = self.layers[i].backward(error_tensor)
        return error_tensor

    def append_layer(self, layer):
        if layer.trainable:
            optimizer_copy = copy.deepcopy(self.optimizer)
            layer.optimizer(optimizer_copy)
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        for i in range(len(self.layers) - 1):
            input_tensor = self.layers[i].forward(input_tensor)
        probabilities = input_tensor
        return probabilities
