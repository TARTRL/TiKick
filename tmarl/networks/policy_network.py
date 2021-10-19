
import torch
import torch.nn as nn

from tmarl.networks.utils.util import init, check
from tmarl.networks.utils.mlp import MLPBase, MLPLayer
from tmarl.networks.utils.rnn import RNNLayer
from tmarl.networks.utils.act import ACTLayer
from tmarl.networks.utils.popart import PopArt
from tmarl.utils.util import get_shape_from_obs_space

# networks are defined here

class PolicyNetwork(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(PolicyNetwork, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal 
        self._activation_id = args.activation_id
        self._use_policy_active_masks = args.use_policy_active_masks 
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_influence_policy = args.use_influence_policy
        self._influence_layer_N = args.influence_layer_N 
        self._use_policy_vhead = args.use_policy_vhead 
        self._recurrent_N = args.recurrent_N 
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)

        self._mixed_obs = False
        self.base = MLPBase(args, obs_shape, use_attn_internal=False, use_cat_self=True)
        
        input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(input_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            input_size = self.hidden_size

        if self._use_influence_policy:
            self.mlp = MLPLayer(obs_shape[0], self.hidden_size,
                              self._influence_layer_N, self._use_orthogonal, self._activation_id)
            input_size += self.hidden_size

        self.act = ACTLayer(action_space, input_size, self._use_orthogonal, self._gain)

        if self._use_policy_vhead:
            init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
            def init_(m): 
                return init(m, init_method, lambda x: nn.init.constant_(x, 0))
            if self._use_popart:
                self.v_out = init_(PopArt(input_size, 1, device=device))
            else:
                self.v_out = init_(nn.Linear(input_size, 1))

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        actor_features = self.base(obs)
        
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)

        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        
        actor_features = self.base(obs)
        
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, available_actions, active_masks = active_masks if self._use_policy_active_masks else None)

        values = self.v_out(actor_features) if self._use_policy_vhead else None
       
        return action_log_probs, dist_entropy, values

    def get_policy_values(self, obs, rnn_states, masks):        
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.base(obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        
        values = self.v_out(actor_features)

        return values