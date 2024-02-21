from multilayer_perceptron import MultiLayerPerceptron


def main():
    mlp = MultiLayerPerceptron()
    mlp.add(25)
    mlp.add(32, activation_function="relu")
    mlp.add(16, activation_function="relu")
    mlp.add(4, activation_function="softmax")


if __name__ == "__main__":
    main()
