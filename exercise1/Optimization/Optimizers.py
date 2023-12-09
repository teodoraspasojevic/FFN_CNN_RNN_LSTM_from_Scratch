import numpy as np


class Sgd:
    """
    Optimizer that updates the weight tensor using stochastic gradient descent.

    Attributes:
        learning_rate(float): Step with which we calculate the update.
    """
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Updates weight tensor.

        Args:
            weight_tensor(np.ndarray): Tensor with weights of the layer.
            gradient_tensor(np.ndarray): Tensor with calculated gradient values for the layer.

        Returns:
            np.ndarray: Tensor with updated weight for the layer.
        """
        updated_weight_tensor = weight_tensor - self.learning_rate * gradient_tensor

        return updated_weight_tensor


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.momentum = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.momentum = self.momentum_rate * self.momentum - self.learning_rate * gradient_tensor
        updated_weight_tensor = weight_tensor + self.momentum
        return updated_weight_tensor


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.momentum = 0
        self.second_momentum = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.momentum = self.mu * self.momentum + (1 - self.mu) * gradient_tensor
        self.second_momentum = self.rho * self.second_momentum + (1 - self.rho) * gradient_tensor * gradient_tensor
        updated_weight_tensor = weight_tensor - self.learning_rate * self.momentum / (np.sqrt(self.second_momentum) + np.finfo.eps)
        return updated_weight_tensor
