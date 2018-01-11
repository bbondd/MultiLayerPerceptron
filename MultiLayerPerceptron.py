from enum import Enum
import math


def sigmoid(x):
    return 1 / (1 + (math.e ** (-x)))


class NodeType(Enum):
    Normal = 1
    Bias = 2


class Node(object):
    def __init__(self, node_type):
        self.nodeType = node_type

        if node_type is NodeType.Bias:
            self.net = 1
            self.out = 1
            self.delta = 1

        else:
            self.net = None
            self.out = None
            self.delta = None

        self.net_connected_nodes = []
        self.out_connected_nodes = []

    def get_net(self, weights):
        if self.net is not None:
            return self.net

        temp_sum = 0
        for node in self.net_connected_nodes:
            temp_sum += node.get_out(weights) * weights[(node, self)]

        self.net = temp_sum

        return self.net

    def get_out(self, weights):
        if self.out is not None:
            return self.out

        self.out = sigmoid(self.get_net(weights))

        return self.out

    def get_delta(self, weights):
        if self.delta is not None:
            return self.delta

        if not self.out_connected_nodes:    # if self is output node
            self.delta = (self.out - self.expect_out) * self.out * (1 - self.out)
            return self.delta

        temp_sum = 0
        for node in self.out_connected_nodes:
            temp_sum += node.get_delta(weights) * weights[(self, node)]

        self.delta = temp_sum
        return self.delta


class Layer(object):
    def __init__(self, node_number):
        self.nodes = []
        for _ in range(node_number):
            self.nodes.append(Node(NodeType.Normal))
        self.nodes.append(Node(NodeType.Bias))

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
    def __init__(self, input_layer_node_number, hidden_layer_node_number, hidden_layer_number, output_layer_node_number, initial_weight):
        self.layers = []
        self.layers.append(Layer(input_layer_node_number))
        for _ in range(hidden_layer_number):
            self.layers.append(Layer(hidden_layer_node_number))
        self.layers.append(Layer(output_layer_node_number))

        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]
        self.output_layer.nodes.pop(-1)   # remove bias node of output layer

        self.weights = {}
        for i in range(len(self.layers) - 1):
            self.connect_two_layers(self.layers[i], self.layers[i + 1], initial_weight)

    def connect_two_layers(self, layer_a, layer_b, initial_weight):
        for node_a in layer_a.nodes:
            for node_b in layer_b.nodes:
                if node_b.nodeType is not NodeType.Bias:
                    node_b.net_connected_nodes.append(node_a)
                    node_a.out_connected_nodes.append(node_b)
                    self.weights.update({(node_a, node_b): initial_weight})

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

    def update_weights(self, input_data, expect_output, learning_constant):
        self.input_layer.set_nets(input_data)
        self.output_layer.get_outs(self.weights)
        self.output_layer.set_expect_outs(expect_output)
        self.input_layer.get_deltas(self.weights)

        for (node_a, node_b) in self.weights:
            self.weights[(node_a, node_b)] -= node_a.out * node_b.delta * learning_constant

        self.reset_nodes()


a = Perceptron(2, 3, 3, 1, 0)

for _ in range(100):
    a.update_weights([0, 0], [0], 0.5)

for _ in range(100):
    a.update_weights([1, 1], [1], 0.5)

print(a.get_result([0, 0]))
print(a.get_result([0, 1]))
print(a.get_result([1, 0]))
print(a.get_result([1, 1]))


