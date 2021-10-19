import torch

from tmarl.networks.policy_network import PolicyNetwork

class MAPPOModule:
    def __init__(self, args, obs_space, share_obs_space, act_space, device=torch.device("cpu")):
        
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.act_space = act_space

        self.actor = PolicyNetwork(args, self.obs_space, self.act_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    def get_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, deterministic=False):
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)

        return None, actions, action_log_probs, rnn_states_actor, None