from . Base import BaseLayer
from scipy.signal import correlate
import numpy as np
import copy


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.trainable = True
        self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))
        self.bias = np.random.uniform(0, 1, num_kernels)
        self._gradient_weights = None
        self._gradient_bias = None
        self.padding_shape = None
        self._weight_optimizer = None
        self._bias_optimizer = None

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    def forward(self, input_tensor):

        if len(input_tensor.shape) == 3:
            input_tensor = np.expand_dims(input_tensor, axis=3)
            expanded = True
        else:
            expanded = False
        b, c, y, x = input_tensor.shape

        if len(self.convolution_shape) == 2:
            convolution_shape = self.convolution_shape + (1,)
        else:
            convolution_shape = self.convolution_shape

        if len(self.stride_shape) == 1:
            self.stride_shape = self.stride_shape * 2

        if self.weights.ndim == 3:
            weights = np.expand_dims(self.weights, axis=-1)
        else:
            weights = self.weights

        output_tensor = np.zeros([b, self.num_kernels, y, x])

        for i in range(b):
            for k in range(self.num_kernels):
                for channel in range(c):
                    output_tensor[i, k, :, :] += correlate(input_tensor[i, channel, :, :], weights[k, channel, :, :], mode="same", method="direct")
                output_tensor[i, k, :, :] += self.bias[k]
        output_tensor = output_tensor[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]

        if expanded:
            output_tensor = np.squeeze(output_tensor, axis=-1)

        return output_tensor

    @property
    def optimizer(self):
        return self._weight_optimizer, self._bias_optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._weight_optimizer = copy.deepcopy(optimizer)
        self._bias_optimizer = copy.deepcopy(optimizer)

    def initialize(self, weight_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])
        self.weights = weight_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, self.convolution_shape[0], self.num_kernels)
