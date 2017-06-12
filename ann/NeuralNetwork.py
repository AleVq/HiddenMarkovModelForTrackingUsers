import numpy as np
import math


# sigmoid function and its derivative
def sigmoid(x):
    result = 1.0 / (1.0 + np.power(math.e, -x))
    return result


def sigmoid_deriv(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


# applying threshold to output layer, x is an array
def theta(x):
    for i in range(0, x.shape[0]):
        if x[i] > 0.5:
            x[i] = 1
        else:
            x[i] = 0
    return x


# softmax function, to apply on the output of the net
def softmax(out):
    exponentials = np.exp(out - out.max())
    return exponentials / np.sum(exponentials)


# cross entropy function,
# inputs: two monodimensional vectors,
# output: cross entropy distance
def cross_entropy(targets, prediction):
    logs = np.log(prediction)
    result = np.array(targets * np.log(prediction))
    result = np.sum(result)
    return - result


class NeuralNetwork:

    # defining the basic structure of the NN: a list of matrices
    # the i-th matrix represents all weights
    # between the i-th layer and the (i+1)-th layer
    def __init__(self, nodes_per_layer):
        # nodesPerLayer = array in which the i-th element
        # corresponds to the i-th layer and gives the number of neurons in that layer
        self.errors = []
        self.activ_func = sigmoid
        self.activ_func_der = sigmoid_deriv
        self.weights = []
        # each matrix has m rows, where m = num of previous layer's nodes,
        # and n columns, where n = num of next layer's nodes
        # values are initiated randomly
        for i in np.arange(1, nodes_per_layer.shape[0]):
            if i == nodes_per_layer.shape[0]-1:  # no bias in the output layer
                temp = 2*np.random.random((nodes_per_layer[i - 1] + 1, nodes_per_layer[i])) - 1
            else:
                temp = 2*np.random.random((nodes_per_layer[i - 1] + 1, nodes_per_layer[i] + 1)) - 1
            self.weights.append(temp)

    # learning with back propagation
    # x = training set, t = example's target
    def train(self, x, t, learning_rate, runs):
        x.as_matrix()
        t = np.array(t)
        # adding the bias to the input layer
        biases = np.atleast_2d(np.ones(x.shape[0]))
        x = np.concatenate((biases.T, x), axis=1)
        for run in range(runs):
            i = np.random.randint(x.shape[0])  # get an example randomly from the training set
            temp = []
            for n in x[i]:
                if n == np.NaN:
                    temp.append(0)
                else:
                    temp.append(n)
            layers_output = [temp]
            # feed forward up to the output layer (included)
            for layer in range(len(self.weights)):
                dot_product = np.dot(layers_output[layer], self.weights[layer])
                activation = self.activ_func(dot_product)
                layers_output.append(activation)
            # TODO applying softmax
            layers_output[-1] = softmax(layers_output[-1])
            # total error of the net
            err = cross_entropy(t[i], layers_output[-1])
            # starting collecting all deltas grouped by layer
            deltas = [err * self.activ_func_der(layers_output[-1])]
            # determine delta for all nodes,
            # from the last hidden layer to the first one:
            for layer in range(len(layers_output)-2, 0, -1):
                dot_prod = deltas[-1].dot(self.weights[layer].T)
                layer_derivate = self.activ_func_der(layers_output[layer])
                to_append = dot_prod * layer_derivate
                deltas.append(to_append)
            # ordering deltas from first hidden layer to output layer
            deltas.reverse()
            # updating weights for each "weight-layer"
            for weight_layer in range(len(self.weights)):
                layer = np.atleast_2d(layers_output[weight_layer])
                delta = np.atleast_2d(deltas[weight_layer])
                self.weights[weight_layer] += learning_rate * layer.T.dot(delta) # delta rule

    # returns the error of the network w.r.t. the target's value of a single instance
    def test(self, x, target):
        last_layer_output = [np.concatenate(([1], x))]
        for weight_layer in range(0, len(self.weights)):
            last_layer_output = self.activ_func(np.dot(last_layer_output, self.weights[weight_layer]))
        error = cross_entropy(target, last_layer_output)
        return error

    # returns the error of the network w.r.t. the target's value of a single instance
    # after applying a threshold on the output layer
    def test_theta(self, x, target):
        last_layer_output = np.concatenate(([1], x))
        for weight_layer in range(0, len(self.weights)):
            last_layer_output = self.activ_func(np.dot(last_layer_output, self.weights[weight_layer]))
        # TODO applying softmax function
        last_layer_output = softmax(last_layer_output)
        predicted_positive_elem = np.argmax(last_layer_output)
        threshold_applied = np.zeros((last_layer_output.shape[0]))  # theta(last_layer_output[-1])
        threshold_applied[predicted_positive_elem] = 1
        result = threshold_applied - target
        return result

