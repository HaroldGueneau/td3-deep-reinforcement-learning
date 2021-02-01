from collections import deque
import random
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VanillaMemory:

    def __init__(self, memory_size, batch_size):
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, done):
        e = [state, action, reward, next_state, done]
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)      

        states = [e[0] for e in experiences]
        actions = [e[1] for e in experiences]
        rewards = [e[2] for e in experiences]
        next_states = [e[3] for e in experiences]
        dones = [e[4] for e in experiences]

        states = self.__convert_to_tensor(states)
        actions = self.__convert_to_tensor(actions)
        rewards = self.__convert_to_tensor(rewards)
        next_states = self.__convert_to_tensor(next_states)
        dones = self.__convert_to_tensor(dones)

        return (states, actions, rewards, next_states, dones)

    def __convert_to_tensor(self, input_list):
        output_tensor = torch.tensor([]).to(device)
        for element in input_list:
            output_tensor = torch.cat((output_tensor, element), dim=0)
        return output_tensor

    def __len__(self):
        return len(self.memory)
