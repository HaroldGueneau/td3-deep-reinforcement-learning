from copy import deepcopy
import numpy as np
import pandas as pd
import itertools

from agent.td3_agent import AgentTD3
from agent.agent_trainer import AgentTrainer
from configurator.multi_layers_perceptron_configurator import MultiLayersPerceptronConfigurator
from configurator.agent_configurator import AgentConfigurator


class GridCalibrator:

    def __init__(self, env, nb_iterations: int, nb_episodes: int, nb_time_steps: int, moving_average_period: int, possibilities: dict):
        self.env = deepcopy(env)
        self.nb_iterations = nb_iterations
        self.possibilities = possibilities
        self.agent_trainer = AgentTrainer(nb_episodes=nb_episodes, max_time_step=nb_time_steps, moving_average_period=50, save_checkpoints=False)

    def calibrate_agent(self):
        grid = self.__build_grid(self.possibilities)
        grid_dict = grid.to_dict('index')
        grid['mean_score'] = 0.0
        row = 0
        for params_combi in grid_dict.values():
            print('----------------------------------------------------')
            print('hyperparameters combination', row, ":", params_combi)
            score_sum = 0
            for i in range(self.nb_iterations):
                print('iteration', i+1)
                agent = self.__initialize_an_agent(hyperparameters=params_combi)
                scores, ma_score = self.agent_trainer.train(agent, self.env)
                score_sum += np.mean(ma_score)
            grid.loc[row, 'mean_score'] = score_sum/self.nb_iterations
            row += 1
        return grid


    def __initialize_an_agent(self, hyperparameters: dict):
        state_size = 3
        action_size = 1
        actor_config = MultiLayersPerceptronConfigurator(
                                                        input_size = state_size,
                                                        output_size = action_size, 
                                                        hidden_layers = [64, 128, 64],
                                                        learning_rate = hyperparameters['actor_learning_rate'], 
                                                        hidden_activation = 'elu', 
                                                        activation_output = 'tanh'
                                                        )

        critic_1_config = MultiLayersPerceptronConfigurator(
                                                            input_size = state_size + action_size,
                                                            output_size = 1, 
                                                            hidden_layers = [64, 128, 64],
                                                            learning_rate = hyperparameters['critic_1_learning_rate'], 
                                                            hidden_activation = 'elu', 
                                                            activation_output = 'linear'
                                                            )

        critic_2_config = MultiLayersPerceptronConfigurator(
                                                            input_size = state_size + action_size,
                                                            output_size = 1, 
                                                            hidden_layers = [64, 128, 64], 
                                                            learning_rate = hyperparameters['critic_2_learning_rate'], 
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
        return AgentTD3(agent_config, actor_config, critic_1_config, critic_2_config)

    def __build_grid(self, possibilities: dict):
        every_possibilities = list(possibilities.values())
        every_possibilities = list(itertools.product(*every_possibilities))
        columns_grid = list(possibilities.keys())
        grid = pd.DataFrame(index=range(len(every_possibilities)), columns=columns_grid)
        for row in grid.index:
            i = 0
            for param_name in list(possibilities.keys()):
                grid.loc[row, param_name] = every_possibilities[row][i]
                i += 1
        return grid