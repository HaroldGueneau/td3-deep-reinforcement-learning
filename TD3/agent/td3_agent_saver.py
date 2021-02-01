import torch

from agent.td3_agent import AgentTD3


class AgentTD3Saver:
    
    def save(self, agent: AgentTD3, id_model: str):
        torch.save(agent.actor_local.state_dict(), 'TD3/output/model_' + str(id_model) +'_weights_actor.pth')
        torch.save(agent.critic_local1.state_dict(), 'TD3/output/model_' + str(id_model) +'_weights_critic1.pth')
        torch.save(agent.critic_local2.state_dict(), 'TD3/output/model_' + str(id_model) +'_weights_critic2.pth')
