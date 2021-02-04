from utils.activation import Activation


class MultiLayersPerceptronConfigurator:
    
    def __init__(self, input_size: int, output_size: int, hidden_layers: list, learning_rate: float, hidden_activation: Activation, activation_output: Activation):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.hidden_activation = hidden_activation
        self.activation_output = activation_output
        self.__check_inputs()
        
    def __check_inputs(self):
        self.__check_input_size()
        self.__check_output_size()
        self.__check_hidden_layers()
        self.__check_learning_rate()
        self.__check_hidden_activation()
        self.__check_activation_output()
        
    def __check_input_size(self):
        if not isinstance(self.input_size, int):
            raise TypeError("The input_size must be an int")
        if self.input_size < 0:
            raise ValueError("The input_size must be a positive value")
            
    def __check_output_size(self):
        if not isinstance(self.output_size, int):
            raise TypeError("The output_size must be an int")
        if self.output_size < 0:
            raise ValueError("The output_size must be a positive value")
       
    def __check_hidden_layers(self):
        if not isinstance(self.hidden_layers, list):
            raise TypeError("The hidden_layers must be a list")
        for layer_size in self.hidden_layers:
            if not isinstance(layer_size, int):
                raise TypeError("Each layer_size in the hidden_layers list must be an int")
            if layer_size < 0:
                raise ValueError("Each layer_size must be a positive value")
    
    def __check_learning_rate(self):
        if not isinstance(self.learning_rate, float):
            raise TypeError("The learning_rate must be an float")
        if self.learning_rate > 1.0 or self.learning_rate < 0.0:
            raise ValueError("The learning_rate must be in [0.0, 1.0]")
            
    def __check_hidden_activation(self):
        if not isinstance(self.hidden_activation, Activation):
            raise TypeError("The hidden_activation must be an Activation Enum")
        
    def __check_activation_output(self):
        if not isinstance(self.activation_output, Activation):
            raise TypeError("The activation_output must be an Activation Enum")
