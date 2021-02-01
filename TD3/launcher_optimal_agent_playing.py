import gym
import numpy as np
import torch
from collections import deque

from configurator.multi_layers_perceptron_configurator import MultiLayersPerceptronConfigurator
from configurator.agent_configurator import AgentConfigurator
from agent.td3_agent import AgentTD3
from agent.td3_agent_loader import AgentTD3Loader
from utils.preprocessor import Preprocessor

NB_EPISODE = 100


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = gym.make('Pendulum-v0')

# For the Pendulum-v0
state_size = 3
action_size = 1

actor_config = MultiLayersPerceptronConfigurator(
                                                input_size = state_size,
                                                output_size = action_size, 
                                                hidden_layers = [64, 128, 64], 
                                                learning_rate = 1e-3, 
                                                hidden_activation = 'elu', 
                                                activation_output = 'tanh'
                                                )

critic_1_config = MultiLayersPerceptronConfigurator(
                                                    input_size = state_size + action_size,
                                                    output_size = 1, 
                                                    hidden_layers = [64, 128, 64], 
                                                    learning_rate = 1e-2, 
                                                    hidden_activation = 'elu', 
                                                    activation_output = 'linear'
                                                    )

critic_2_config = MultiLayersPerceptronConfigurator(
                                                    input_size = state_size + action_size,
                                                    output_size = 1, 
                                                    hidden_layers = [64, 128, 64], 
                                                    learning_rate = 1e-2, 
                                                    hidden_activation = 'elu', 
                                                    activation_output = 'linear'
                                                    )

agent_config = AgentConfigurator(
                                memory_size = 100000,
                                batch_size = 128, 
                                gamma = 0.99, 
                                tau = 1e-3, 
                                action_min = -2.0, 
                                action_max = 2.0,
                                freq_update_actor = 5
                                )

agent = AgentTD3(agent_config, actor_config, critic_1_config, critic_2_config)
loader = AgentTD3Loader()
agent = loader.load_optimal_agent(agent)

preprocessor = Preprocessor(device)

ma_score = deque(maxlen=100)
scores = []

for i_episode in range(1, NB_EPISODE+1):
    state = env.reset()
    score_episode = 0
    for t in range(200):
        action = agent.choose_an_action(preprocessor.preprocess_one_numpy_state(state))
        env.render()
        state, reward, done, _ = env.step(preprocessor.preprocess_pytorch_action(action))
        state = state.reshape((3))
        score_episode += reward[0]
        if done:
            break 
    ma_score.append(score_episode)
    scores.append(score_episode)
    print('\rEpisode {}\tAverage Score over the last {} episodes: {:.2f}, Last score: {}'.format(i_episode, 100, np.mean(ma_score), score_episode))

