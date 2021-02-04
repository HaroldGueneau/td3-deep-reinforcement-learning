from enum import Enum


class Activation(Enum):
    """
    An Enum that contains all possible activations
    """

    linear = 1
    tanh = 2
    relu = 3
    leaky_relu = 4
    elu = 5
