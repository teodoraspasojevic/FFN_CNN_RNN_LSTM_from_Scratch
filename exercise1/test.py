import unittest
from Layers import *
from Optimization import *
import numpy as np
from scipy import stats
from scipy.ndimage.filters import gaussian_filter
import NeuralNetwork
import matplotlib.pyplot as plt
import os
import tabulate
import argparse


class L2Loss:

    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        return np.sum(np.square(input_tensor - label_tensor))

    def backward(self, label_tensor):
        return 2*np.subtract(self.input_tensor, label_tensor)


class TestPooling(unittest.TestCase):
    plot = False
    directory = 'plots/'

    def setUp(self):
        self.batch_size = 2
        self.input_shape = (2, 4, 7)
        self.input_size = np.prod(self.input_shape)

        np.random.seed(1337)
        self.input_tensor = np.random.uniform(-1, 1, (self.batch_size, *self.input_shape))

        self.categories = 12
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

        self.layers = list()
        self.layers.append(None)
        self.layers.append(Flatten.Flatten())
        self.layers.append(L2Loss())
        self.plot_shape = (self.input_shape[0], np.prod(self.input_shape[1:]))

    def test_trainable(self):
        layer = Pooling.Pooling((2, 2), (2, 2))
        self.assertFalse(layer.trainable, "Possible reason: Pooling doesn't inherit from the base layer.")

    def test_shape(self):
        layer = Pooling.Pooling((2, 2), (2, 2))
        result = layer.forward(self.input_tensor)
        expected_shape = np.array([self.batch_size, 2, 2, 3])
        self.assertEqual(np.sum(np.abs(np.array(result.shape) - expected_shape)), 0,
                         "Possible reason: Output tensor from forward pass in Pooling has the wrong shape. Make sure to"
                         "calculate the correct shape in case of even and odd dimensions.")

    def test_overlapping_shape(self):
        layer = Pooling.Pooling((2, 1), (2, 2))
        result = layer.forward(self.input_tensor)
        expected_shape = np.array([self.batch_size, 2, 2, 6])
        self.assertEqual(np.sum(np.abs(np.array(result.shape) - expected_shape)), 0,
                         "Possible reason: Output tensor from forward pass in Pooling has the wrong shape. Make sure to"
                         "include the stride in both dimensions into your computations.")

    def test_subsampling_shape(self):
        layer = Pooling.Pooling((3, 2), (2, 2))
        result = layer.forward(self.input_tensor)
        expected_shape = np.array([self.batch_size, 2, 1, 3])
        self.assertEqual(np.sum(np.abs(np.array(result.shape) - expected_shape)), 0,
                         "Possible reason: Output tensor from forward pass in Pooling has the wrong shape. Make sure to"
                         "include the stride in both dimensions into your computations.")

    def test_gradient_stride(self):
        self.layers[0] = Pooling.Pooling((2, 2), (2, 2))
        difference = Helpers.gradient_check(self.layers, self.input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-6,
                             "Possible reason: If the tests for the forward pass fail as well, fix those first. If they"
                             "pass, your backward pass is not correct. Make sure, you write the error back to the"
                             "correct index, that you stored in the forward pass. It might help, to do this upsampling"
                             "on paper and compare your intermediate results to the ones from the backward pass.")

    def test_gradient_overlapping_stride(self):
        label_tensor = np.random.random((self.batch_size, 24))
        self.layers[0] = Pooling.Pooling((2, 1), (2, 2))
        difference = Helpers.gradient_check(self.layers, self.input_tensor, label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-6,
                             "Possible reason: If the tests for the forward pass fail as well, fix those first. If they"
                             "pass, your backward pass is not correct. Make sure, you write the error back to the"
                             "correct index, that you stored in the forward pass. It might help, to do this upsampling"
                             "on paper and compare your intermediate results to the ones from the backward pass.")

    def test_gradient_subsampling_stride(self):
        label_tensor = np.random.random((self.batch_size, 6))
        self.layers[0] = Pooling.Pooling((3, 2), (2, 2))
        difference = Helpers.gradient_check(self.layers, self.input_tensor, label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-6,
                             "Possible reason: If the tests for the forward pass fail as well, fix those first. If they"
                             "pass, your backward pass is not correct. Make sure, you write the error back to the"
                             "correct index, that you stored in the forward pass. It might help, to do this upsampling"
                             "on paper and compare your intermediate results to the ones from the backward pass.")

    def test_layout_preservation(self):
        pool = Pooling.Pooling((1, 1), (1, 1))
        input_tensor = np.array(range(np.prod(self.input_shape) * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = pool.forward(input_tensor)
        self.assertAlmostEqual(np.sum(np.abs(output_tensor-input_tensor)), 0.,
                               "Possible reason: For a pooling region of 1x1 and a stride of 1 the values in your the"
                               "input_tensor get changed. Check, if you store the selected maximum at the correct place"
                               "of the output_tensor.")

    def test_expected_output_valid_edgecase(self):
        input_shape = (1, 3, 3)
        pool = Pooling.Pooling((2, 2), (2, 2))
        batch_size = 2
        input_tensor = np.array(range(np.prod(input_shape) * batch_size), dtype=float)
        input_tensor = input_tensor.reshape(batch_size, *input_shape)
        result = pool.forward(input_tensor)
        expected_result = np.array([[[[4]]], [[[13]]]])
        self.assertEqual(np.sum(np.abs(result - expected_result)), 0,
                         "Possible reason: In the forward pass the pooling region doesn't start at [0,0] or is not"
                         "shifted according to the stride. It might help to do the pooling for this input on paper and"
                         "compare your intermediate values to the ones from the forward pass.")

    def test_expected_output(self):
        input_shape = (1, 4, 4)
        pool = Pooling.Pooling((2, 2), (2, 2))
        batch_size = 2
        input_tensor = np.array(range(np.prod(input_shape) * batch_size), dtype=float)
        input_tensor = input_tensor.reshape(batch_size, *input_shape)
        result = pool.forward(input_tensor)
        expected_result = np.array([[[[5.,  7.], [13., 15.]]], [[[21., 23.], [29., 31.]]]])
        self.assertEqual(np.sum(np.abs(result - expected_result)), 0,
                         "Possible reason: In the forward pass the wrong values are selected for the maximum or the"
                         "pooling region lies over the wrong areas. It might help to do the pooling for this input"
                         "on paper and compare your intermediate values to the ones from the forward pass.")


if __name__ == "__main__":
    import unittest

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPooling)

    runner = unittest.TextTestRunner()
    runner.run(suite)
