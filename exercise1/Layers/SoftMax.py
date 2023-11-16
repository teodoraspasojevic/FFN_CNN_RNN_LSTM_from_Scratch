from . Base import BaseLayer
import numpy as np


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output_tensor = None

    def forward(self, input_tensor):
        input_tensor_shifted = input_tensor - np.max(input_tensor, axis=1).reshape(-1, 1)
        exp_tensor = np.exp(input_tensor_shifted)
        sums = np.sum(exp_tensor, axis=1)
        sums = np.tile(sums, (input_tensor.shape[1], 1)).transpose()
        probabilities = exp_tensor / sums
        self.output_tensor = probabilities
        return probabilities

    def backward(self, error_tensor):
        # TODO: check this function
        inner_sum = np.sum(error_tensor * self.output_tensor, axis=1, keepdims=True)
        previous_error_tensor = self.output_tensor * (error_tensor - inner_sum)
        return previous_error_tensor
