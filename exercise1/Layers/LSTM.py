import numpy as np
from . Base import BaseLayer
from . FullyConnected import FullyConnected
from . Sigmoid import Sigmoid
from . TanH import TanH
import copy


class LSTM(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = 0

        self.cell_state = np.zeros(shape=(1, self.hidden_size))
        self.hidden_state = np.zeros(shape=(1, self.hidden_size))

        self.fc_gates = None
        self.fc_output = None
        self.sigmoid_forget_gate = None
        self.sigmoid_input_gate = None
        self.sigmoid_output_gate = None
        self.sigmoid_final_output = None
        self.tanh_cell_gate = None
        self.tanh_hidden = None
        self.create_embedded_layers()

        self.gate_inputs = None
        self.output_inputs = np.zeros(shape=(self.batch_size, self.hidden_size + 1))
        self.sigmoid_forget_gate_activations = None
        self.sigmoid_input_gate_activations = None
        self.sigmoid_output_gate_activations = None
        self.sigmoid_final_output_activations = None
        self.tanh_cell_gate_activations = None
        self.tanh_hidden_activations = None

        self.current_cell_error = np.zeros(shape=(1, self.hidden_size))
        self.current_hidden_error = np.zeros(shape=(1, self.hidden_size))

        self.forget_states = np.zeros(shape=(self.batch_size, self.hidden_size))
        self.input_states = np.zeros(shape=(self.batch_size, self.hidden_size))
        self.output_states = np.zeros(shape=(self.batch_size, self.hidden_size))
        self.candidate_cell_states = np.zeros(shape=(self.batch_size, self.hidden_size))
        self.cell_states = np.zeros(shape=(self.batch_size, self.hidden_size))
        self.hidden_inputs = np.zeros(shape=(self.batch_size, self.hidden_size))

        # self._weights = np.random.uniform(0, 1, size=(self.hidden_size + self.input_size + 1, self.hidden_size * 4))
        # self.weights_output = np.random.uniform(0, 1, size=(self.hidden_size + 1, self.output_size))
        self._gradient_weights = None
        self.gradient_weights_output = None

        self.input_tensor = None
        self._optimizer = None
        self._memorize = False

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer_object):
        self._optimizer = optimizer_object

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, flag):
        self._memorize = flag

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weight):
        self._gradient_weights = gradient_weight

    @property
    def weights(self):
        return self.fc_gates.weights

    @weights.setter
    def weights(self, weight):
        self.fc_gates.weights = weight

    def create_embedded_layers(self):
        self.fc_gates = FullyConnected(self.hidden_size + self.input_size, self.hidden_size * 4)
        self.fc_output = FullyConnected(self.hidden_size, self.output_size)
        self.sigmoid_forget_gate = Sigmoid()
        self.sigmoid_input_gate = Sigmoid()
        self.sigmoid_output_gate = Sigmoid()
        self.sigmoid_final_output = Sigmoid()
        self.tanh_cell_gate = TanH()
        self.tanh_hidden = TanH()

    def initialize(self, weight_initializer, bias_initializer):
        fan_in = self.hidden_size + self.input_size + 1
        fan_out = self.hidden_size
        weights = weight_initializer.initialize(self.fc_gates.weights.shape, fan_in, fan_out)
        self.fc_gates.weights = weights
        fan_in = self.hidden_size + 1
        fan_out = self.output_size
        weights_output = weight_initializer.initialize(self.fc_output.weights.shape, fan_in, fan_out)
        self.fc_output.weights = weights_output

    def calculate_regularization_loss(self):
        return self.optimizer.regularizer.norm(self.weights)

    def forward(self, input_tensor):

        self.input_tensor = input_tensor
        self.batch_size = input_tensor.shape[0]

        sigmoid_forget_gate_activations = np.zeros(shape=(self.batch_size, self.hidden_size))
        sigmoid_input_gate_activations = np.zeros(shape=(self.batch_size, self.hidden_size))
        sigmoid_output_gate_activations = np.zeros(shape=(self.batch_size, self.hidden_size))
        sigmoid_final_output_activations = np.zeros(shape=(self.batch_size, self.output_size))
        tanh_cell_gate_activations = np.zeros(shape=(self.batch_size, self.hidden_size))
        tanh_hidden_activations = np.zeros(shape=(self.batch_size, self.hidden_size))

        gate_inputs = np.zeros(shape=(self.batch_size, self.input_size + self.hidden_size + 1))
        output_inputs = np.zeros(shape=(self.batch_size, self.hidden_size + 1))
        output_tensor = np.zeros(shape=(self.batch_size, self.output_size))

        forget_states = np.zeros(shape=(self.batch_size, self.hidden_size))
        input_states = np.zeros(shape=(self.batch_size, self.hidden_size))
        output_states = np.zeros(shape=(self.batch_size, self.hidden_size))
        candidate_cell_states = np.zeros(shape=(self.batch_size, self.hidden_size))
        cell_states = np.zeros(shape=(self.batch_size, self.hidden_size))
        hidden_inputs = np.zeros(shape=(self.batch_size, self.hidden_size))

        if not self.memorize:
            self.hidden_state = np.zeros(shape=(1, self.hidden_size))
            self.cell_state = np.zeros(shape=(1, self.hidden_size))

        for i in range(self.batch_size):

            # Prepare input for the first pass.
            current_input_tensor = input_tensor[i, :]
            current_input_tensor = np.transpose(current_input_tensor.reshape(-1, 1))
            current_input_tensor = np.concatenate((self.hidden_state, current_input_tensor), axis=1)

            # Compute outputs of the gates.
            gate = self.fc_gates.forward(current_input_tensor)

            forget_gate_input = gate[:, self.hidden_size]
            input_gate_input = gate[:, self.hidden_size:self.hidden_size * 2]
            output_gate_input = gate[:, self.hidden_size * 2: self.hidden_size * 3]
            cell_gate_input = gate[:, self.hidden_size * 3:]

            forget_state = self.sigmoid_forget_gate.forward(forget_gate_input)
            input_state = self.sigmoid_input_gate.forward(input_gate_input)
            output_state = self.sigmoid_output_gate.forward(output_gate_input)
            candidate_cell_state = self.tanh_cell_gate.forward(cell_gate_input)

            # Compute cell state.
            cell_state = forget_state * self.cell_state + input_state * candidate_cell_state
            self.cell_state = cell_state

            # Compute hidden state.
            hidden_input = self.tanh_hidden.forward(cell_state)
            hidden_state = output_state * hidden_input
            self.hidden_state = hidden_state

            # Compute output state.
            output = self.sigmoid_final_output.forward(self.fc_output.forward(hidden_state))
            output_tensor[i, :] = output

            # Save activations.
            sigmoid_forget_gate_activations[i, :] = self.sigmoid_forget_gate.activations
            sigmoid_input_gate_activations[i, :] = self.sigmoid_input_gate.activations
            sigmoid_output_gate_activations[i, :] = self.sigmoid_output_gate.activations
            tanh_cell_gate_activations[i, :] = self.tanh_cell_gate.activations
            tanh_hidden_activations[i, :] = self.tanh_hidden.activations
            sigmoid_final_output_activations[i, :] = self.sigmoid_final_output.activations

            # Save inputs for FC layers.
            gate_inputs[i, :] = self.fc_gates.input_tensor
            output_inputs[i, :] = self.fc_output.input_tensor

            # Save outputs of the gates.
            forget_states[i, :] = forget_state
            input_states[i, :] = input_state
            output_states[i, :] = output_state
            candidate_cell_states[i, :] = candidate_cell_state
            cell_states[i, :] = cell_state
            hidden_inputs[i, :] = hidden_input

        self.gate_inputs = gate_inputs
        self.output_inputs = output_inputs
        self.sigmoid_forget_gate_activations = sigmoid_forget_gate_activations
        self.sigmoid_input_gate_activations = sigmoid_input_gate_activations
        self.sigmoid_output_gate_activations = sigmoid_output_gate_activations
        self.sigmoid_final_output_activations = sigmoid_final_output_activations
        self.tanh_cell_gate_activations = tanh_cell_gate_activations
        self.tanh_hidden_activations = tanh_hidden_activations
        self.forget_states = forget_states
        self.input_states = input_states
        self.output_states = output_states
        self.candidate_cell_states = candidate_cell_states
        self.cell_states = cell_states
        self.hidden_inputs = hidden_inputs

        return output_tensor

    def backward(self, error_tensor):

        accumulated_weights_gradient = np.zeros_like(self.weights)
        accumulated_weights_output_gradient = np.zeros_like(self.fc_output.weights)

        previous_error_tensor = np.zeros_like(self.input_tensor)
        self.current_hidden_error = 0
        self.current_cell_error = 0

        gradients = list()

        for i in reversed(range(len(self.input_tensor))):

            # Set the activations of the activation layers.
            current_activation = self.sigmoid_forget_gate_activations[i, :]
            current_activation = np.transpose(current_activation.reshape(-1, 1))
            self.sigmoid_forget_gate.activations = current_activation

            current_activation = self.sigmoid_input_gate_activations[i, :]
            current_activation = np.transpose(current_activation.reshape(-1, 1))
            self.sigmoid_input_gate.activations = current_activation

            current_activation = self.sigmoid_output_gate_activations[i, :]
            current_activation = np.transpose(current_activation.reshape(-1, 1))
            self.sigmoid_output_gate.activations = current_activation

            current_activation = self.tanh_cell_gate_activations[i, :]
            current_activation = np.transpose(current_activation.reshape(-1, 1))
            self.tanh_cell_gate.activations = current_activation

            current_activation = self.tanh_hidden_activations[i, :]
            current_activation = np.transpose(current_activation.reshape(-1, 1))
            self.tanh_hidden.activations = current_activation

            current_activation = self.sigmoid_final_output_activations[i, :]
            current_activation = np.transpose(current_activation.reshape(-1, 1))
            self.sigmoid_final_output.activations = current_activation

            # Set the inputs of the FC layers.
            current_gate_input = self.gate_inputs[i, :]
            current_gate_input = np.transpose(current_gate_input.reshape(-1, 1))
            self.fc_gates.input_tensor = current_gate_input

            current_output_input = self.output_inputs[i, :]
            current_output_input = np.transpose(current_output_input.reshape(-1, 1))
            self.fc_output.input_tensor = current_output_input

            # Propagate error from yt to ht.
            previous_error = self.sigmoid_final_output.backward(error_tensor[i])
            previous_error = self.fc_output.backward(previous_error)
            previous_error += self.current_hidden_error                                # BP of copy

            # Propagate from ht to [ht-1, xt].
            previous_error_o_t = previous_error * self.hidden_inputs[i, :]             # BP of *
            previous_error_o_t = self.sigmoid_output_gate.backward(previous_error_o_t)

            # Propagate from ht to ct.
            previous_error_c_t = previous_error * self.output_states[i, :]             # BP of *
            previous_error_c_t = self.tanh_hidden.backward(previous_error_c_t)
            previous_error_c_t += self.current_cell_error                              # BP of copy

            # Propagate error from ct to addition elements 1 and 2.
            previous_error_add1 = previous_error_c_t                                    # BP of +
            previous_error_add2 = previous_error_c_t                                    # BP of +

            # Propagate error from addition element 2 to [ht-1, xt].
            previous_error_c_tilda_t = previous_error_add2 * self.input_states[i, :]    # BP of *
            previous_error_c_tilda_t = self.tanh_cell_gate.backward(previous_error_c_tilda_t)

            previous_error_i_t = previous_error_add2 * self.candidate_cell_states[i, :]  # BP of *
            previous_error_i_t = self.sigmoid_input_gate.backward(previous_error_i_t)

            # Propagate from addition element 1.
            previous_error_c_t_1 = previous_error_add1 * self.forget_states[i, :]        # BP of *

            if i != 0:
                previous_error_f_t = previous_error_add1 * self.cell_states[i, :]
            else:
                previous_error_f_t = previous_error_add1 * np.zeros(shape=(1, self.hidden_size))
            previous_error_f_t = self.sigmoid_forget_gate.backward(previous_error_f_t)

            # Propagate through FC_gates.
            previous_error = np.concatenate((previous_error_f_t, previous_error_i_t, previous_error_o_t, previous_error_c_tilda_t), axis=1)
            previous_error = self.fc_gates.backward(previous_error)

            # Unpack all gradient.
            gradient_weights = self.fc_gates.gradient_weights
            gradient_output = self.fc_output.gradient_weights

            gradients.append(gradient_weights)

            # Accumulate the gradients.
            accumulated_weights_gradient += gradient_weights
            accumulated_weights_output_gradient += gradient_output

            # Save gradient w.r.t. ht-1 for the computations for sample t-1 (we need it for BP for copy function).
            self.current_hidden_error = previous_error[:, :self.hidden_size]

            # Save gradient w.r.t. ct-1 for the computations for sample t-1 (we need it for BP for copy function).
            self.current_cell_error = previous_error_c_t_1

            # Save error tensor.
            previous_error_tensor[i, :] = previous_error[:, self.hidden_size:self.hidden_size + self.input_size]

        self.gradient_weights = accumulated_weights_gradient
        self.gradient_weights_output = accumulated_weights_output_gradient

        # Update weights in FC layers.
        if self.optimizer:
            updated_weight_tensor = self.optimizer.calculate_update(self.fc_gates.weights, accumulated_weights_gradient)
            # self.weights = updated_weight_tensor
            self.fc_gates.weights = updated_weight_tensor

        if self.optimizer:
            updated_weight_tensor = self.optimizer.calculate_update(self.fc_output.weights, accumulated_weights_output_gradient)
            self.fc_output.weights = updated_weight_tensor
            # self.weights_output = updated_weight_tensor

        return previous_error_tensor
