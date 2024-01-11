from . Base import BaseLayer
import numpy as np


class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.dropout_mask = None

    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor
        else:
            self.dropout_mask = np.random.binomial(1/self.probability, self.probability, size=input_tensor.shape)
            return self.dropout_mask * input_tensor

    def backward(self, error_tensor):
        return error_tensor * self.dropout_mask
