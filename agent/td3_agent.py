import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from utils.multi_layers_perceptron import MultiLayerPerceptron
from agent.vanilla_memory import VanillaMemory
from configurator.multi_layers_perceptron_configurator import MultiLayersPerceptronConfigurator
from configurator.agent_configurator import AgentConfigurator
from utils.activation import Activation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AgentTD3():

    def __init__(self, agent_config: AgentConfigurator, actor_config: MultiLayersPerceptronConfigurator, critic_1_config: MultiLayersPerceptronConfigurator, critic_2_config: MultiLayersPerceptronConfigurator):
        self.agent_config = agent_config
        
        self.__initialize_neural_networks(actor_config, critic_1_config, critic_2_config)
        self.complete_update_of_target_models()
        self.__set_eval_mode()

        self.learning_step_count = 0 # Learning step counter (how many times we learned)
        self.freq_update_actor = agent_config.freq_update_actor # Frequency we will update the actor
        self.memory = VanillaMemory(self.agent_config.memory_size, self.agent_config.batch_size)

    def __initialize_neural_networks(self, actor_config: MultiLayersPerceptronConfigurator, critic_1_config: MultiLayersPerceptronConfigurator, critic_2_config: MultiLayersPerceptronConfigurator):
        self.actor_local = MultiLayerPerceptron(actor_config).to(device)
        self.actor_target = MultiLayerPerceptron(actor_config).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=actor_config.learning_rate)
        self.activation_output_actor =  actor_config.activation_output

        self.critic_local1 = MultiLayerPerceptron(critic_1_config).to(device)
        self.critic_target1 = MultiLayerPerceptron(critic_1_config).to(device)
        self.critic_optimizer1 = optim.Adam(self.critic_local1.parameters(), lr=critic_1_config.learning_rate)

        self.critic_local2 = MultiLayerPerceptron(critic_2_config).to(device)
        self.critic_target2 = MultiLayerPerceptron(critic_2_config).to(device)
        self.critic_optimizer2 = optim.Adam(self.critic_local2.parameters(), lr=critic_2_config.learning_rate)

    def complete_update_of_target_models(self):
        self.__soft_update(self.critic_local1, self.critic_target1, 1)
        self.__soft_update(self.critic_local2, self.critic_target2, 1)
        self.__soft_update(self.actor_local, self.actor_target, 1)  

    def __soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def __set_eval_mode(self):
        self.actor_local.eval()
        self.actor_target.eval()
        self.critic_local1.eval()
        self.critic_target1.eval()
        self.critic_local2.eval()
        self.critic_target2.eval()

    def tackle_a_step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > self.agent_config.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.agent_config.gamma)

    def choose_an_action(self, state):
        with torch.no_grad():
            action = self.actor_local(state).cpu()
        action += torch.clamp(torch.tensor(np.random.normal(scale=0.2)), -0.5, 0.5) # gaussian noise in [-0.5, 0.5]
        action = self.__resquale_action(action) # To have actions in the env's allowed interval
        action = torch.clamp(action, self.agent_config.action_min, self.agent_config.action_max)
        return action

    def __resquale_action(self, action):
        if self.activation_output_actor == Activation.tanh:
            actual_min = -1
            actual_max = 1
        else:
            raise ValueError("The allowed activations for the actor are: Activation.tanh")
        action = (action - actual_min) / (actual_max - actual_min)
        action = action * (self.agent_config.action_max - self.agent_config.action_min) + self.agent_config.action_min
        return action

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        self.__update_critics(states, actions, rewards, next_states, dones, gamma)

        if self.learning_step_count % self.freq_update_actor==0:
            self.__update_actor(states)

        self.__soft_update(self.critic_local1, self.critic_target1, self.agent_config.tau)
        self.__soft_update(self.critic_local2, self.critic_target2, self.agent_config.tau)
        self.__soft_update(self.actor_local, self.actor_target, self.agent_config.tau)  
        
        self.learning_step_count += 1

    # Critic Update

    def __update_critics(self, states, actions, rewards, next_states, dones, gamma):
        # Activate the training mode for critic networks
        self.critic_local1.train()
        self.critic_local2.train()
        # Training Process
        q_target = self.__compute_q_target(next_states, rewards, gamma, dones)
        critic_loss = self.__compute_critic_loss(states, actions, q_target)
        self.__minimize_critic_loss(critic_loss)
        # Desactivate the training mode for critic networks
        self.critic_local1.eval()
        self.critic_local2.eval()

    def __compute_q_target(self, next_states, rewards, gamma, dones):
        with torch.no_grad():
            # Get predicted next-state actions with noise
            actions_next = self.actor_target(next_states)
            actions_next += torch.clamp(torch.tensor(np.random.normal(scale=0.2)), -0.5, 0.5) # gaussian noise in [-0.5, 0.5]
            actions_next = torch.clamp(actions_next, self.agent_config.action_min, self.agent_config.action_max)
            # Get Q values from target models
            input_q_target_next = torch.cat((next_states, actions_next), dim=1).to(device)
            q_targets_next1 = self.critic_target1(input_q_target_next)
            q_targets_next2 = self.critic_target2(input_q_target_next)
            # Compute Q targets for current states (y_i)
            q_targets1 = rewards + (gamma * q_targets_next1 * (1 - dones))
            q_targets2 = rewards + (gamma * q_targets_next2 * (1 - dones))
            q_target = torch.minimum(q_targets1, q_targets2)
        return q_target

    def __compute_critic_loss(self, states, actions, q_target):
        # Compute critic expected Q value (local models)
        input_q_expected = torch.cat((states, actions), dim=1).to(device)
        q_expected1 = self.critic_local1(input_q_expected)
        q_expected2 = self.critic_local2(input_q_expected)
        # Compute critic loss
        critic_loss1 = F.mse_loss(q_expected1, q_target)
        critic_loss2 = F.mse_loss(q_expected2, q_target)
        critic_loss =  critic_loss1 + critic_loss2
        return critic_loss

    def __minimize_critic_loss(self, critic_loss):
        self.critic_optimizer1.zero_grad()
        self.critic_optimizer2.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local1.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.critic_local2.parameters(), 1)
        self.critic_optimizer1.step()
        self.critic_optimizer2.step()

    # Actor Update

    def __update_actor(self, states):
        self.actor_local.train() # Activate the training mode for the actor network
        actor_loss = self.__compute_actor_loss(states)
        self.__minimize_actor_loss(actor_loss)
        self.actor_local.eval() # Desactivate the training mode for the actor network

    def __compute_actor_loss(self, states):
        actions_pred = self.actor_local(states)
        actions_pred = self.__resquale_action(actions_pred)
        input_actor_loss = torch.cat((states, actions_pred), dim=1).to(device)
        actor_loss = -self.critic_local1(input_actor_loss).mean()
        return actor_loss

    def __minimize_actor_loss(self, actor_loss):
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
