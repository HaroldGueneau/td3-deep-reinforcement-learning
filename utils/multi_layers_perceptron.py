import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from configurator.multi_layers_perceptron_configurator import MultiLayersPerceptronConfigurator
from utils.layer import Layer


class MultiLayerPerceptron(nn.Module):

    def __init__(self, mlp_configurator: MultiLayersPerceptronConfigurator):
        super(MultiLayerPerceptron, self).__init__()
        self.hidden_layers = self.__build_hidden_layers(hidden_layers=mlp_configurator.hidden_layers, 
                                                        hidden_activation=mlp_configurator.hidden_activation, 
                                                        input_size=mlp_configurator.input_size)
        self.output_layer = Layer(input_size=mlp_configurator.hidden_layers[-1], 
                                    output_size=mlp_configurator.output_size, 
                                    activation=mlp_configurator.activation_output)

    def __build_hidden_layers(self, hidden_layers: list, hidden_activation: str, input_size: int):
        hidden_layers_modules = nn.ModuleList([])
        previous_hidden_size = input_size
        for hidden_size in hidden_layers:
            hidden_layers_modules.append(Layer(input_size=previous_hidden_size, 
                                                output_size=hidden_size, 
                                                activation=hidden_activation))
            previous_hidden_size = hidden_size
        return hidden_layers_modules

    def forward(self, x):
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        return self.output_layer(x)
