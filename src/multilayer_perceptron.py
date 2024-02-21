import numpy as np
from typing import List
import random
from layer import Layer
from utils import ActivationFunctions


class MultiLayerPerceptron:

    """Class representing a MultiLayer Perceptron and its training and prediction methods."""

    def __init__(self):
        self.layers: List[Layer] = []

    def add(self, layerSize, activation_function=None):
        """Add a new layer to the MLP"""

        n = len(self.layers)
        if n == 0:
            # input layer has no activation function
            newLayer = Layer(layerSize, activation_function=None)
        else:
            prevLayerSize = self.layers[n - 1].size
            newLayer = Layer(layerSize, prevLayerSize, activation_function)

        self.layers.append(newLayer)

    def forward_pass(self, input: np.ndarray) -> List[np.ndarray]:
        """Perform forward pass through the network."""

        self.layers[0].activations = input
        zs = []
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            prev_layer = self.layers[i - 1]
            z = np.dot(layer.weights, prev_layer.activations) + layer.biases
            zs.append(z)
            newActivations = layer.activation_function(z)

            layer.activations = newActivations

        return zs

    def backpropagate(self, target: int, zs: List[np.ndarray],
                      bias_deltas: List[np.ndarray], weight_deltas: List[np.ndarray]) -> float:
        """Run back-propagation using Stochastic Gradient Descent to update weights and biases."""

        ideal = np.zeros((4, 1), dtype=float)
        ideal[target] = 1

        cost = ActivationFunctions.cross_entropy_loss(
            self.layers[-1].activations, ideal)

        delta = ActivationFunctions.cross_entropy_cost_derivative(
            self.layers[-1].activations, ideal)
        bias_deltas[-1] = delta
        weight_deltas[-1] = np.dot(delta,
                                   self.layers[-2].activations.reshape(1, -1))

        for l in range(2, len(self.layers)):
            z = zs[-l]
            sp = self.layers[-l].activation_function_derivative(z)
            delta = np.dot(
                self.layers[-l + 1].weights.transpose(), delta) * sp
            bias_deltas[-l] = delta
            weight_deltas[-l] = np.dot(delta,
                                       self.layers[-l - 1].activations.reshape(1, -1))

        return cost

    def train(self, data, targets, epochs=100, learning_rate=0.001, early_stopping_patience=None,
              exp_decay_rate=None, decay_step_in_epochs=None, batch_size=10, verbose=False):
        """Train the MLP with the provided data and targets over a number of epochs."""

        dataset = list(zip(data, targets))
        min_average_cost = float('inf')
        epochs_since_min = 0
        print("Beginning training.")
        for i in range(epochs):
            if early_stopping_patience and epochs_since_min > early_stopping_patience:  # Simple early stopping
                if verbose:
                    print("Early stopping: no improvement in 10 epochs.")
                break

            current_learning_rate = learning_rate
            if exp_decay_rate and decay_step_in_epochs:
                current_learning_rate = learning_rate * \
                    (exp_decay_rate ** (i / decay_step_in_epochs))

            total_cost = 0

            random.shuffle(dataset)
            batches = [dataset[k:k+batch_size]
                       for k in range(0, len(dataset), batch_size)]  # Split data into batches
            for batch in batches:
                batch_weight_deltas = [None for _ in self.layers]
                batch_bias_deltas = [None for _ in self.layers]
                for index, (datum, target) in enumerate(batch):
                    datum_weight_deltas = [None for _ in self.layers]
                    datum_bias_deltas = [None for _ in self.layers]
                    zs = self.forward_pass(np.array(datum).reshape(-1, 1))
                    cost = self.backpropagate(
                        target, zs, datum_bias_deltas, datum_weight_deltas)
                    total_cost += cost

                    # Update batch deltas
                    batch_weight_deltas = [np.add(batch_weight_deltas[i], datum_weight_deltas[i]) if batch_weight_deltas[i]
                                           is not None else datum_weight_deltas[i] for i in range(len(self.layers))]
                    batch_bias_deltas = [np.add(batch_bias_deltas[i], datum_bias_deltas[i]) if batch_bias_deltas[i]
                                         is not None else datum_bias_deltas[i] for i in range(len(self.layers))]

                for index, layer in enumerate(self.layers):
                    if index == 0:
                        continue

                    # Update weights and biases
                    layer.update_weights(
                        batch_weight_deltas[index], current_learning_rate)
                    layer.update_biases(
                        batch_bias_deltas[index], current_learning_rate)

            average_cost = total_cost / len(data)
            if early_stopping_patience:
                if round(average_cost, 3) < round(min_average_cost, 3):  # Early stopping on min loss
                    min_average_cost = average_cost
                    epochs_since_min = 0

                    # store optimal weights and biases
                    optimal_weights = [layer.weights for layer in self.layers]
                    optimal_biases = [layer.biases for layer in self.layers]
                else:
                    epochs_since_min += 1

            if verbose:
                print(
                    f"Epoch {i}:")
                print(
                    f"accuracy: {self.evaluate(zip(data, targets))} / {len(data)} - loss: {round(average_cost, 3)}")

        if early_stopping_patience:
            # restore optimal weights and biases
            for index, layer in enumerate(self.layers):
                layer.weights = optimal_weights[index]
                layer.biases = optimal_biases[index]

        if verbose:
            print(
                f"Final accuracy: {self.evaluate(zip(data, targets))} / {len(data)}")
            if early_stopping_patience:
                print(f"Final loss: {min_average_cost}")
            else:
                print(f"Final loss: {average_cost}")

        print("Training complete.")

    def evaluate(self, test_data):
        """Evaluate the model on a given test set and return the number of correct predictions."""

        results = []
        for datum, target in test_data:
            self.forward_pass(np.array(datum).reshape(-1, 1))
            results.append((np.argmax(self.layers[-1].activations), target))

        return sum(int(x == y) for x, y in results)

    def predict(self, datum):
        """Predict the class of a single input."""

        self.forward_pass(np.array(datum).reshape(-1, 1))
        return np.argmax(self.layers[-1].activations)
