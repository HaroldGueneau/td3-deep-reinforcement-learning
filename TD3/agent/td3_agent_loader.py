import torch

from agent.td3_agent import AgentTD3


class AgentTD3Loader:
    
    def load(self, agent: AgentTD3, id_model: str):
        agent.actor_local.load_state_dict(torch.load('TD3/output/model_' + str(id_model) +'_weights_actor.pth'))
        agent.critic_local1.load_state_dict(torch.load('TD3/output/model_' + str(id_model) +'_weights_critic1.pth'))
        agent.critic_local2.load_state_dict(torch.load('TD3/output/model_' + str(id_model) +'_weights_critic2.pth'))
        # Copy local parameters
        agent.complete_update_of_target_models()
        return agent

    def load_optimal_agent(self, agent: AgentTD3):
        agent.actor_local.load_state_dict(torch.load('TD3/output/optimal_agent/weights_actor.pth'))
        agent.critic_local1.load_state_dict(torch.load('TD3/output/optimal_agent/weights_critic1.pth'))
        agent.critic_local2.load_state_dict(torch.load('TD3/output/optimal_agent/weights_critic2.pth'))
        # Copy local parameters
        agent.complete_update_of_target_models()
        return agent
