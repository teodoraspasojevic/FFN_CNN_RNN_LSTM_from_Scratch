from . Base import BaseLayer
from . FullyConnected import FullyConnected
from . Sigmoid import Sigmoid
from . TanH import TanH
import numpy as np

import sys

# Add the absolute path of the 'DeepLearningFAU' directory to sys.path
sys.path.append('C:\Projects\DL\Deep-Learning-Assignments')
from exercise1.Optimization.Optimizers import Adam


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(self.hidden_size)
        self._memorize = False
        self._weights = np.random.uniform(0, 1, size=(self.hidden_size + self.input_size + 1, self.hidden_size))
        self.weights_output = np.random.uniform(0, 1, size=(self.output_size + 1, self.hidden_size))
        self.fc_hidden = None
        self.fc_output = None
        self.sigmoid = None
        self.tanh = None
        self.create_embedded_layers()
        self.input_tensor = None
        self.current_hidden_gradient = 0

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

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        output_tensor = np.zeros((self.input_tensor.shape[0], self.output_size))

        if not self.memorize:
            self.hidden_state = np.zeros(self.hidden_size)

        for i in range(self.input_tensor.shape[0]):

            input_hidden = np.concatenate((self.hidden_state, self.input_tensor[i, :]), axis=0)
            input_hidden = np.append(input_hidden, 1)
            input_hidden = np.transpose(input_hidden.reshape(-1, 1))

            # Computing hidden state
            hidden_state = self.tanh.forward(self.fc_hidden.forward(input_hidden))
            self.hidden_state = hidden_state

            # Computing output state
            input_output = np.append(self.hidden_state, 1)
            output_tensor[i, :] = self.sigmoid.forward(self.fc_output.forward(input_output))

        return output_tensor

    def backward(self, error_tensor):

        accumulated_error = 0
        accumulated_weights_gradient = 0
        accumulated_output_gradient = 0

        for i in range(self.input_tensor[0]-1, 0):

            # Propagate error from yt to xt.
            previous_error_tensor = self.sigmoid.backward(error_tensor[i])
            previous_error_tensor = self.fc_output.backward(previous_error_tensor)
            previous_error_tensor += self.current_hidden_gradient                         # BP of copy
            previous_error_tensor = self.tanh.backward(previous_error_tensor)
            previous_error_tensor = self.fc_hidden.backward(previous_error_tensor)

            # Unpack all gradient.
            gradient_weights = self.gradient_weights
            weights_hh_gradient = gradient_weights[0:self.hidden_size]

            gradient_output = self.fc_output.gradient_weights

            # Save gradient w.r.t. ht for the computations for sample t-1 (we need it for BP for copy function).
            self.current_hidden_gradient = weights_hh_gradient

            # Accumulate gradient w.r.t. input that will be returned.
            accumulated_error += previous_error_tensor

            # Accumulate gradient
            accumulated_weights_gradient += gradient_weights
            accumulated_output_gradient += gradient_output

        # Updated weights in FC layers.
        optimizer = Adam(1e-3, 0.9, 0.999)

        self.fc_hidden.optimizer(optimizer)
        self.fc_hidden.optimize(accumulated_weights_gradient)
        self.fc_hidden.optimizer(None)

        self.fc_output.optimizer(optimizer)
        self.fc_output.optimize(accumulated_output_gradient)
        self.fc_output.optimizer(None)

        # Set weights of RNN to the updated values.
        self.weights(self.fc_hidden.weights)
        self.weights_output = self.fc_output.weights

        return accumulated_error
