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
        jacobian_matrix = np.dot(self.output_tensor, (np.eye(self.output_tensor.shape[1]) - self.output_tensor.transpose()))
        previous_error_tensor = np.dot(error_tensor, jacobian_matrix)
        return previous_error_tensor
