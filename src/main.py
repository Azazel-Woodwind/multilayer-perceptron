from multilayer_perceptron import MultiLayerPerceptron


def main():
    print("Creating Neural Network...")
    mlp = MultiLayerPerceptron()
    mlp.add(25)
    mlp.add(32, activation_function="relu")
    mlp.add(16, activation_function="relu")
    mlp.add(4, activation_function="softmax")

    data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    target = [0, 1, 1, 0]
    mlp.train(data, target, epochs=100, early_stopping_patience=10, learning_rate=0.01,
              exp_decay_rate=0.9, decay_step_in_epochs=5, batch_size=10, verbose=True)

    print("Predicting...")
    print(mlp.predict([0, 0]))


if __name__ == "__main__":
    main()
