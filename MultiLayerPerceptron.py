from enum import Enum
import math
import random


def sigmoid(x):
    return 1 / (1 + (math.e ** (-x)))


class NodeType(Enum):
    Normal = 1
    Bias = 2


class Node(object):
    def __init__(self, node_type):
        self.nodeType = node_type


class LayerType(Enum):
    Input = 1
    Hidden = 2
    Output = 3


class Layer(object):
    def __init__(self, layer_type, node_number):
        self.layerType = layer_type
        self.nodes = []
        for i in range(node_number):
            self.nodes.append(Node(NodeType.Normal))

        if self.layerType is not LayerType.Output:
            self.nodes.append(Node(NodeType.Bias))


class Perceptron(object):
    def __init__(self, input_layer_node_number, hidden_layer_node_number, hidden_layer_number, output_layer_number):
        self.inputLayer = Layer(LayerType.Input, input_layer_node_number)
        self.hiddenLayers = []
        for i in range(hidden_layer_number):
            self.hiddenLayers.append(Layer(LayerType.Hidden), hidden_layer_node_number)
        self.outputLayer = Layer(LayerType.Output, output_layer_number)

