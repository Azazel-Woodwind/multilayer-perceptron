import numpy as np


class ActivationFunctions:

    """Class containing strictly static methods for activation functions and their derivatives."""

    @staticmethod
    def cross_entropy_loss(q, y):
        return -np.sum(y * np.log(q))

    @staticmethod
    def cross_entropy_cost_derivative(activations, targets):
        return activations - targets

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(z):
        temp = ActivationFunctions.sigmoid(z)
        return temp * (1 - temp)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return e_x / e_x.sum(axis=0, keepdims=True)
