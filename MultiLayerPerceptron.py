from enum import Enum
import math
import random


def sigmoid(x, derivative=False):
    if derivative is True:
        return x * (1 - x)
    else:
        return 1 / (1 + (math.e ** (-x)))


def rectified_linear_unit(x, derivative=False):
    if derivative is True:
        if x >= 0:
            return 1
        else:
            return 0
    else:
        return max(0.0, x)


def leaky_rectified_linear_unit(x, derivative=False):
    if derivative is True:
        if x > 0:
            return 1
        else:
            return 0.01

    else:
        return max(0.01 * x, x)


def identity(x, derivative=False):
    if derivative is True:
        return 1
    else:
        return x


def tan_h(x, derivative=False):
    if derivative is True:
        return 1 / (x ** 2 + 1)
    else:
        return 2 / (1 + math.e ** (-2 * x)) - 1


def inverse_square_root_linear_unit(x, derivative=False):
    if derivative is True:
        if x < 0:
            return (1 + x ** 2) ** (-1.5)
        else:
            return 1
    else:
        if x < 0:
            return x * ((1 + x ** 2) ** (-0.5))
        else:
            return x


class NodeType(Enum):
    Normal = 1
    Bias = 2


class Node(object):
    def __init__(self, node_type, activation_function):
        self.nodeType = node_type
        self.activationFunction = activation_function

        if node_type is NodeType.Bias:
            self.net = 1
            self.out = 1
            self.delta = 1

        else:
            self.net = None
            self.out = None
            self.delta = None

        self.netConnectedNodes = []
        self.outConnectedNodes = []

    def get_net(self, weights):
        if self.net is not None:
            return self.net

        temp_sum = 0
        for node in self.netConnectedNodes:
            temp_sum += node.get_out(weights) * weights[(node, self)]

        self.net = temp_sum

        return self.net

    def get_out(self, weights):
        if self.out is not None:
            return self.out

        self.out = self.activationFunction(self.get_net(weights))
        return self.out

    def get_delta(self, weights):
        if self.delta is not None:
            return self.delta

        if not self.outConnectedNodes:    # if self is output node
            self.delta = (self.out - self.expect_out) * self.activationFunction(self.out, True)
            return self.delta

        temp_sum = 0
        for node in self.outConnectedNodes:
            temp_sum += node.get_delta(weights) * weights[(self, node)]

        self.delta = temp_sum * self.activationFunction(self.out, True)
        return self.delta


class Layer(object):
    def __init__(self, node_number, activation_function):
        self.nodes = []
        for _ in range(node_number):
            self.nodes.append(Node(NodeType.Normal, activation_function))
        self.nodes.append(Node(NodeType.Bias, activation_function))

    def set_nets(self, nets):
        for i in range(len(nets)):
            self.nodes[i].net = nets[i]

    def set_expect_outs(self, expect_outs):
        for i in range(len(expect_outs)):
            self.nodes[i].expect_out = expect_outs[i]

    def get_outs(self, weights):
        outs = []
        for node in self.nodes:
            outs.append(node.get_out(weights))

        return outs

    def get_deltas(self, weights):
        deltas = []
        for node in self.nodes:
            deltas.append(node.get_delta(weights))

        return deltas


class Perceptron(object):
    def __init__(self, layer_node_numbers, hidden_layer_number, activation_function):
        input_layer_node_number = layer_node_numbers[0]
        hidden_layer_node_number = layer_node_numbers[1]
        output_layer_node_number = layer_node_numbers[2]

        self.layers = []
        self.layers.append(Layer(input_layer_node_number, activation_function))
        for _ in range(hidden_layer_number):
            self.layers.append(Layer(hidden_layer_node_number, activation_function))
        self.layers.append(Layer(output_layer_node_number, activation_function))

        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]
        self.output_layer.nodes.pop(-1)   # remove bias node of output layer

        self.weights = {}
        for i in range(len(self.layers) - 1):
            self.connect_two_layers(self.layers[i], self.layers[i + 1])

    def connect_two_layers(self, layer_a, layer_b):
        for node_a in layer_a.nodes:
            for node_b in layer_b.nodes:
                if node_b.nodeType is not NodeType.Bias:
                    node_b.netConnectedNodes.append(node_a)
                    node_a.outConnectedNodes.append(node_b)
                    self.weights.update({(node_a, node_b): random.uniform(-0.2, 0.2)})

    def reset_nodes(self):
        for layer in self.layers:
            for node in layer.nodes:
                if node.nodeType is not NodeType.Bias:
                    node.net = None
                    node.out = None
                    node.delta = None

    def get_result(self, input_data):
        self.input_layer.set_nets(input_data)
        result = self.output_layer.get_outs(self.weights)
        self.reset_nodes()

        return result

    def update_weights(self, input_data, expect_output, learning_rate):
        self.input_layer.set_nets(input_data)
        self.output_layer.get_outs(self.weights)
        self.output_layer.set_expect_outs(expect_output)
        self.input_layer.get_deltas(self.weights)

        for (node_a, node_b) in self.weights:
            self.weights[(node_a, node_b)] -= node_a.out * node_b.delta * learning_rate

        self.reset_nodes()

    def train(self, input_data_set, expect_output_set, learning_rate):
        for i in range(len(input_data_set)):
            self.update_weights(input_data_set[i], expect_output_set[i], learning_rate)
    

def main():
    a = Perceptron([2, 3, 1], 2, leaky_rectified_linear_unit)

    def half_circle(x, y):
        return math.sqrt(x ** 2 + y ** 2)

    input_data_set = []
    expect_output_set = []

    for _ in range(100):
        x = random.random()
        y = random.random()
        input_data_set.append([x, y])
        expect_output_set.append([half_circle(x, y)])

    num = 100000
    learning_rate = 0.3
    for i in range(num):
        a.train(input_data_set, expect_output_set, learning_rate)
        if i % 1000 == 0:
            print(i / num)

    for _ in range(10):
        x = random.random()
        y = random.random()
        print([x, y])
        print(a.get_result([x, y]))


f = open("asdf.txt", 'w')
f.write("hi!")
f.close()

