import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.activation import Activation


class Layer(nn.Module):

    def __init__(self, input_size: int, output_size: int, activation: Activation):
        super(Layer, self).__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=output_size)
        self.activation = self.__set_activation(activation)

    def __set_activation(self, activation: str):
        if activation == Activation.linear:
            return None
        elif activation == Activation.tanh:
            return nn.Tanh()
        elif activation == Activation.relu:
            return nn.ReLU()
        elif activation == Activation.leaky_relu:
            return nn.LeakyReLU()
        elif activation == Activation.elu:
            return nn.ELU()
        else:
            raise ValueError('The activation is not valid.')

    def forward(self, x):
        x = self.linear(x)
        if self.activation != None:
            x = self.activation(x)
        return x.clone()