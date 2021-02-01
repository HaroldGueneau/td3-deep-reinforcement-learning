class AgentConfigurator:
    
    def __init__(self, memory_size: int, batch_size: int, gamma: float, tau: float, action_min: float, action_max: float, freq_update_actor: int):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.action_min = action_min
        self.action_max = action_max
        self.freq_update_actor = freq_update_actor
        self.__check_inputs()
        
    def __check_inputs(self):
        self.__check_memory_size()
        self.__check_batch_size()
        self.__check_gamma()
        self.__check_tau()
        self.__check_action_min()
        self.__check_action_max()
        self.__check_freq_update_actor()
        
    def __check_memory_size(self):
        if not isinstance(self.memory_size, int):
            raise TypeError("The memory_size must be an int")
        if self.memory_size < 0:
            raise ValueError("The memory_size must be a positive value")
    
    def __check_batch_size(self):
        if not isinstance(self.batch_size, int):
            raise TypeError("The batch_size must be an int")
        if self.batch_size < 0:
            raise ValueError("The batch_size must be a positive value")
    
    def __check_gamma(self):
        if not isinstance(self.gamma, float):
            raise TypeError("The gamma must be an float")
        if self.gamma > 1.0 or self.gamma < 0.0:
            raise ValueError("The gamma must be in [0.0, 1.0]")

    def __check_tau(self):
        if not isinstance(self.tau, float):
            raise TypeError("The tau must be an float")
        if self.tau > 1.0 or self.tau < 0.0:
            raise ValueError("The tau must be in [0.0, 1.0]")
    
    def __check_action_min(self):
        if not isinstance(self.action_min, float):
            raise TypeError("The action_min must be an float")

    def __check_action_max(self):
        if not isinstance(self.action_max, float):
            raise TypeError("The action_max must be an float")

    def __check_freq_update_actor(self):
        if not isinstance(self.freq_update_actor, int):
            raise TypeError("The freq_update_actor must be an int")
        if self.freq_update_actor < 0:
            raise ValueError("The freq_update_actor must be a positive value")
        