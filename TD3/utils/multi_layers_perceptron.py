import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from configurator.multi_layers_perceptron_configurator import MultiLayersPerceptronConfigurator


class OurLayer(nn.Module):

    def __init__(self, input_size: int, output_size: int, activation: str):
        super(OurLayer, self).__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=output_size)
        self.activation = self.__set_activation(activation)

    def __set_activation(self, activation: str):
        if activation == 'linear':
            return None
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'elu':
            return nn.ELU()
        else:
            raise ValueError('The activation is not valid.')

    def forward(self, x):
        x = self.linear(x)
        if self.activation != None:
            x = self.activation(x)
        return x.clone()


class MultiLayerPerceptron(nn.Module):

    def __init__(self, mlp_configurator: MultiLayersPerceptronConfigurator):
        super(MultiLayerPerceptron, self).__init__()
        self.hidden_layers = self.__build_hidden_layers(hidden_layers=mlp_configurator.hidden_layers, hidden_activation=mlp_configurator.hidden_activation, input_size=mlp_configurator.input_size)
        self.output_layer = OurLayer(input_size=mlp_configurator.hidden_layers[-1], output_size=mlp_configurator.output_size, activation=mlp_configurator.activation_output)

    def __build_hidden_layers(self, hidden_layers: list, hidden_activation: str, input_size: int):
        hidden_layers_modules = nn.ModuleList([])
        previous_hidden_size = input_size
        for hidden_size in hidden_layers:
            hidden_layers_modules.append(OurLayer(input_size=previous_hidden_size, output_size=hidden_size, activation=hidden_activation))
            previous_hidden_size = hidden_size
        return hidden_layers_modules

    def forward(self, x):
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        return self.output_layer(x)
