import gym
import numpy as np
import torch

from configurator.multi_layers_perceptron_configurator import MultiLayersPerceptronConfigurator
from configurator.agent_configurator import AgentConfigurator
from agent.td3_agent import AgentTD3
from agent.agent_trainer import AgentTrainer
from utils.activation import Activation


env = gym.make('Pendulum-v0')

state_size = 3
action_size = 1

actor_config = MultiLayersPerceptronConfigurator(
                input_size = state_size,
                output_size = action_size, 
                hidden_layers = [64, 128, 64], 
                learning_rate = 1e-3, 
                hidden_activation = Activation.elu,
                activation_output = Activation.tanh
                )

critic_1_config = MultiLayersPerceptronConfigurator(
                    input_size = state_size + action_size,
                    output_size = 1, 
                    hidden_layers = [64, 128, 64], 
                    learning_rate = 1e-2, 
                    hidden_activation = Activation.elu,
                    activation_output = Activation.linear
                    )

critic_2_config = MultiLayersPerceptronConfigurator(
                    input_size = state_size + action_size,
                    output_size = 1, 
                    hidden_layers = [64, 128, 64], 
                    learning_rate = 1e-2, 
                    hidden_activation = Activation.elu,
                    activation_output = Activation.linear
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

agent_trainer = AgentTrainer(nb_episodes=100, max_time_step=200, moving_average_period=10, save_checkpoints=True)
scores, ma_score = agent_trainer.train(agent, env)