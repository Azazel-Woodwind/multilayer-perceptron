import numpy as np
from typing import Optional
from utils import ActivationFunctions


class Layer:

    """Class representing a layer of neurons in a MultiLayer Perceptron."""

    def __init__(self, size: int, prev_layer_size: Optional[int] = None, activation_function: Optional[str] = None):
        self.size = size
        self.weights = np.random.randn(
            size, prev_layer_size) if prev_layer_size else None
        self.biases = np.random.randn(size, 1)
        self.activations = np.zeros((size, 1))
        self.activation_function = getattr(
            ActivationFunctions, activation_function, None) if activation_function else None

    @property
    def activation_function_derivative(self):
        return getattr(ActivationFunctions, f"{self.activation_function.__name__}_derivative", None)

    def update_weights(self, deltas, learning_rate):
        self.weights = self.weights - learning_rate * deltas

    def update_biases(self, deltas, learning_rate):
        self.biases = self.biases - learning_rate * deltas
