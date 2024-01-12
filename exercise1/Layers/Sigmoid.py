from . Base import BaseLayer
import numpy as np


class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.activations = None

    def forward(self, input_tensor):
        self.activations = 1 / (1 + np.exp(-input_tensor))
        return self.activations

    def backward(self, error_tensor):
        gradient_tensor = self.activations * (1 - self.activations)
        previous_error_tensor = gradient_tensor * error_tensor
        return previous_error_tensor
