from . Base import BaseLayer


class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, input_tensor):
        b, w, h, c = input_tensor.shape
        self.input_shape = input_tensor.shape
        input_tensor_reshaped = input_tensor.reshape(b, w * h * c)
        return input_tensor_reshaped

    def backward(self, error_tensor):
        error_tensor_reshaped = error_tensor.reshape(self.input_shape)
        return error_tensor_reshaped
