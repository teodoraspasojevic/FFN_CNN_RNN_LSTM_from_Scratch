import sys
sys.path.append('C:\Projects\DL\Deep-Learning-Assignments')
from exercise1.Optimization.Optimizers import Adam
from . Base import BaseLayer
from . FullyConnected import FullyConnected
from . Sigmoid import Sigmoid
from . TanH import TanH
import numpy as np


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(shape=(1, self.hidden_size))
        self._memorize = False
        self._weights = np.random.uniform(0, 1, size=(self.hidden_size + self.input_size + 1, self.hidden_size))
        self.weights_output = np.random.uniform(0, 1, size=(self.output_size + 1, self.hidden_size))
        self.fc_hidden = None
        self.fc_output = None
        self.sigmoid = None
        self.tanh = None
        self.create_embedded_layers()
        self.input_tensor = None
        self.current_hidden_error = np.zeros(shape=(1, self.hidden_size))
        self.tanh_activations = None
        self.sigmoid_activations = None

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, memorize):
        self._memorize = memorize

    @property
    def gradient_weights(self):
        return self.fc_hidden.gradient_weights

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    def create_embedded_layers(self):
        self.fc_hidden = FullyConnected(self.hidden_size + self.input_size + 1, self.hidden_size)
        self.fc_output = FullyConnected(self.hidden_size + 1, self.output_size)
        self.sigmoid = Sigmoid()
        self.tanh = TanH()

    def initialize(self, weight_initializer, bias_initializer):
        fan_in = self.hidden_size + self.input_size + 1
        fan_out = self.hidden_size
        self.weights = weight_initializer.initialize(self.weights.shape, fan_in, fan_out)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        sigmoid_activations = np.zeros(shape=(self.input_tensor.shape[0], self.output_size))
        tanh_activations = np.zeros(shape=(self.input_tensor.shape[0], self.hidden_size))
        output_tensor = np.zeros(shape=(self.input_tensor.shape[0], self.output_size))

        if not self.memorize:
            self.hidden_state = np.zeros(shape=(1, self.hidden_size))

        for i in range(self.input_tensor.shape[0]):

            # Prepare input for the first pass.
            current_input_tensor = input_tensor[i, :]
            current_input_tensor = np.transpose(current_input_tensor.reshape(-1, 1))
            input_hidden = np.concatenate((self.hidden_state, current_input_tensor), axis=1)
            input_hidden = np.append(input_hidden, np.ones((input_hidden.shape[0], 1)), axis=1)

            # Compute hidden state.
            hidden_state = self.tanh.forward(self.fc_hidden.forward(input_hidden))
            self.hidden_state = hidden_state

            # Compute output state.
            input_output = np.append(self.hidden_state, np.ones(shape=(input_hidden.shape[0], 1)), axis=1)
            output = self.sigmoid.forward(self.fc_output.forward(input_output))
            output_tensor[i, :] = output

            # Save activations.
            tanh_activations[i, :] = hidden_state
            sigmoid_activations[i, :] = output

        self.tanh_activations = tanh_activations
        self.sigmoid_activations = sigmoid_activations

        return output_tensor

    def backward(self, error_tensor):

        accumulated_weights_gradient = 0
        accumulated_output_gradient = 0

        previous_error_tensor = np.zeros_like(self.input_tensor)

        for i in range(self.input_tensor.shape[0]-1, 0, -1):

            # Set the activations of the activation layers.
            current_activation = self.tanh_activations[i, :]
            current_activation = np.transpose(current_activation.reshape(-1, 1))
            self.tanh.activations = current_activation

            current_activation = self.sigmoid_activations[i, :]
            current_activation = np.transpose(current_activation.reshape(-1, 1))
            self.sigmoid.activations = current_activation

            # Propagate error from yt to xt.
            previous_error = self.sigmoid.backward(error_tensor[i])
            previous_error = self.fc_output.backward(previous_error)
            previous_error = previous_error[:, :-1]                                   # remove the bias from tensor
            previous_error += self.current_hidden_error                            # BP of copy
            previous_error = self.tanh.backward(previous_error)
            previous_error = self.fc_hidden.backward(previous_error)

            # Unpack all gradient.
            gradient_weights = self.gradient_weights
            gradient_output = self.fc_output.gradient_weights

            # Accumulate the gradients.
            accumulated_weights_gradient += gradient_weights
            accumulated_output_gradient += gradient_output

            # Save gradient w.r.t. ht for the computations for sample t-1 (we need it for BP for copy function).
            self.current_hidden_error = previous_error[:, 0:self.hidden_size]

            # Save error tensor.
            previous_error_tensor[i, :] = previous_error[:, self.hidden_size:self.hidden_size + self.input_size]

        # Update weights in FC layers.
        optimizer_hidden = Adam(1e-3, 0.9, 0.999)
        self.fc_hidden.optimizer = optimizer_hidden
        self.fc_hidden.optimize(accumulated_weights_gradient)
        self.fc_hidden.optimizer = None

        optimizer_output = Adam(1e-3, 0.9, 0.999)
        self.fc_output.optimizer = optimizer_output
        self.fc_output.optimize(accumulated_output_gradient)
        self.fc_output.optimizer = None

        # Set weights of RNN to the updated values.
        self.weights = self.fc_hidden.weights
        self.weights_output = self.fc_output.weights

        return previous_error_tensor
