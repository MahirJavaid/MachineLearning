import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.weights1 = np.random.rand(2, 2)
        self.biases1 = np.random.rand(2,)
        self.weights2 = np.random.rand(2, 2)
        self.biases2 = np.random.rand(2,)
        self.weights3 = np.random.rand(2, 2)
        self.biases3 = np.random.rand(2,)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return x * (1 - x)

    def predict(self, inputs):
        h1_out = self.sigmoid(np.dot(inputs, self.weights1) + self.biases1)
        h2_out = self.sigmoid(np.dot(h1_out, self.weights2) + self.biases2)
        output = np.round(self.sigmoid(np.dot(h2_out, self.weights3) + self.biases3))
        return output

    def train(self, training_data, labels, learning_rate=10, epochs=1000):
        for epoch in range(epochs):
            for i in range(len(training_data)):
                inputs = training_data[i]
                label = labels[i]

                h1_out = self.sigmoid(np.dot(inputs, self.weights1) + self.biases1)
                h2_out = self.sigmoid(np.dot(h1_out, self.weights2) + self.biases2)
                output = self.sigmoid(np.dot(h2_out, self.weights3) + self.biases3)

                output_error = label - output
                output_delta = output_error * self.sigmoid_prime(output)
                h2_error = output_delta.dot(self.weights3.T)
                h2_delta = h2_error * self.sigmoid_prime(h2_out)
                h1_error = h2_delta.dot(self.weights2.T)
                h1_delta = h1_error * self.sigmoid_prime(h1_out)

                self.weights3 += np.outer(h2_out, output_delta) * learning_rate
                self.biases3 += output_delta * learning_rate
                self.weights2 += np.outer(h1_out, h2_delta) * learning_rate
                self.biases2 += h2_delta * learning_rate
                self.weights1 += np.outer(inputs, h1_delta) * learning_rate
                self.biases1 += h1_delta * learning_rate


training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 0, 0, 1])

my_nn = NeuralNetwork()
my_nn.train(training_data, labels)

test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for inputs in test_inputs:
    output = my_nn.predict(inputs)
    print("Input:", inputs, "Output:", output[0])
