import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Preprocessor:
    def __init__(self, device: str):
        self.device = device
    
    def preprocess_one_numpy_state(self, state):
        return torch.from_numpy(state).float().to(self.device)
    
    def preprocess_pytorch_action(self, action):
        return action.data.numpy()
    
    def preprocess_mdp(self, state, action, reward, next_state, done):
        state = torch.from_numpy(np.array([state])).float().to(device)
        reward = torch.from_numpy(np.array([reward])).float().to(device)
        done = np.array([done])
        done = torch.from_numpy(np.array([done])).float().to(device)
        next_state = next_state.reshape((3))
        action = action.to(device)
        next_state = torch.from_numpy(np.array([next_state])).float().to(device)
        return state, action, reward, next_state, done
