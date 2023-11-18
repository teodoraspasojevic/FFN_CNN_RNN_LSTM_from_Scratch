from . Base import BaseLayer
import numpy as np


class ReLU(BaseLayer):
    """
    Layer implementing ReLU activation function, which reduces "vanishing gradient" problem.

    Attributes:
        trainable(bool): Flag implying if the layer will be trained.
        input_tensor(np.ndarray): Tensor with features of the whole batch. Its size is batch_size * (input_size + 1).
    """
    def __init__(self):
        super().__init__()
        self.input_tensor = None

    def forward(self, input_tensor):
        """
        Calculates the output of the layer in the forward pass, by passing max between input and zero.

        Args:
            input_tensor(np.ndarray): Tensor with inputs for the layer.

        Returns:
            np.ndarray: Output tensor of the layer.
        """
        self.input_tensor = input_tensor
        output_tensor = np.maximum(0, input_tensor)

        return output_tensor

    def backward(self, error_tensor):
        """
        Calculates the error for the previous layer, in the backward pass.

        Args:
            error_tensor(np.ndarray): Tensor with errors the layer after this one generates.

        Returns:
            np.ndarray: Tensor with errors of the layer before this one.
        """
        # We save error value if the input value is greater than zero, else we save zero.
        previous_error_tensor = np.zeros_like(error_tensor)
        previous_error_tensor[self.input_tensor > 0] = error_tensor[self.input_tensor > 0]

        return previous_error_tensor
