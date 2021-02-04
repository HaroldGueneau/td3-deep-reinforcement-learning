from collections import deque
import numpy as np
import torch

from agent.td3_agent import AgentTD3
from agent.td3_agent_saver import AgentTD3Saver
from utils.preprocessor import Preprocessor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AgentTrainer:
    
    def __init__(self, nb_episodes: int, max_time_step: int, moving_average_period: int, save_checkpoints: bool):
        self.nb_episodes = nb_episodes
        self.max_time_step = max_time_step
        self.moving_average_period = moving_average_period
        self.save_checkpoints = save_checkpoints
        self.preprocessor = Preprocessor(device)
        self.saver = AgentTD3Saver()

    def train(self, agent: AgentTD3, env):
        ma_score = deque(maxlen=self.moving_average_period)
        scores = []
        for i_episode in range(1, self.nb_episodes+1):
            scores, ma_score = self.__play_an_episode(env, agent, ma_score, scores, i_episode)
        return scores, ma_score
    
    def __play_an_episode(self, env, agent, ma_score, scores, i_episode):
        state = env.reset()
        score_episode = 0
        for t in range(self.max_time_step):
            state, done, score_episode = self.__tackle_a_step(agent, state, env, score_episode)
            if done:
                break
        ma_score.append(score_episode)
        scores.append(score_episode)
        self.__log_and_save(i_episode, ma_score, agent)
        return scores, ma_score

    def __tackle_a_step(self, agent, state, env, score_episode):
        action = agent.choose_an_action(self.preprocessor.preprocess_one_numpy_state(state))
        next_state, reward, done, _ = env.step(self.preprocessor.preprocess_pytorch_action(action))
        state, action, reward, next_state, done = self.preprocessor.preprocess_mdp(state, action, reward, next_state, done)
        agent.tackle_a_step(state, action, reward, next_state, done)
        state = next_state.cpu().data.numpy()[0]
        score_episode += reward.cpu().data.numpy()[0][0]
        return state, done, score_episode

    def __log_and_save(self, i_episode, ma_score, agent):
        print('\rEpisode {}\tAverage Score over the last {} episodes: {:.2f}'.format(i_episode, self.moving_average_period, np.mean(ma_score)), end="")
        if i_episode % self.moving_average_period == 0:
            print('\rEpisode {}\tAverage Score over the last {} episodes: {:.2f}'.format(i_episode, self.moving_average_period, np.mean(ma_score)))
            if self.save_checkpoints:
                self.saver.save(agent, 'episode_' + str(i_episode))
