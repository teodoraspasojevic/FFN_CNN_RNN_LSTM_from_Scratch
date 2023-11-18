from . Base import BaseLayer
import numpy as np


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(0, 1, size=(input_size + 1, output_size)) # we are adding place for bias in weights
        self._optimizer = None
        self.input_tensor = None
        self.gradient_tensor = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, new_optimizer):
        self._optimizer = new_optimizer

    @property
    def gradient_weights(self):
        # TODO: check if this is correct!
        return self.gradient_tensor

    def forward(self, input_tensor):
        bias_input = np.ones(shape=(input_tensor.shape[0], 1))
        self.input_tensor = np.append(input_tensor, bias_input, axis=1)
        output_tensor = np.dot(self.input_tensor, self.weights)
        return output_tensor

    def backward(self, error_tensor):
        previous_error_tensor = np.dot(error_tensor, self.weights.transpose())
        previous_error_tensor = previous_error_tensor[:, :-1]
        gradient_tensor = np.dot(self.input_tensor.transpose(), error_tensor)
        self.gradient_tensor = gradient_tensor
        if self.optimizer:
            updated_weight_tensor = self.optimizer.calculate_update(self.weights, gradient_tensor)
            self.weights = updated_weight_tensor
        return previous_error_tensor
