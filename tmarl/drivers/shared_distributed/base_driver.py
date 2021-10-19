import numpy as np
import torch

def _t2n(x):
    return x.detach().cpu().numpy()

class Driver(object):
    def __init__(self, config, client=None):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if 'signal' in config:
            self.actor_id = config['signal'].actor_id
            self.weight_ids = config['signal'].weight_ids
        else:
            self.actor_id = 0
            self.weight_ids = [0]

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps if hasattr(self.all_args,'num_env_steps') else self.all_args.eval_num

        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.learner_n_rollout_threads = self.all_args.n_rollout_threads

        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.hidden_size = self.all_args.hidden_size
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir



        if self.algorithm_name == "rmappo":
            from tmarl.algorithms.r_mappo_distributed.mappo_algorithm import MAPPOAlgorithm as TrainAlgo
            from tmarl.algorithms.r_mappo_distributed.mappo_module import MAPPOModule as AlgoModule
        else:
            raise NotImplementedError

        if self.envs:
            share_observation_space = self.envs.share_observation_space[0] \
            if self.use_centralized_V else self.envs.observation_space[0]
            # policy network
            self.algo_module = AlgoModule(self.all_args,
                                          self.envs.observation_space[0],
                                          share_observation_space,
                                          self.envs.action_space[0],
                                          device=self.device)

        else:
            share_observation_space = self.eval_envs.share_observation_space[0] \
                if self.use_centralized_V else self.eval_envs.observation_space[0]
            # policy network
            self.algo_module = AlgoModule(self.all_args,
                                          self.eval_envs.observation_space[0],
                                          share_observation_space,
                                          self.eval_envs.action_space[0],
                                          device=self.device)

        if self.model_dir is not None:
            self.restore()

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.algo_module, device=self.device)


        # buffer
        from tmarl.replay_buffers.normal.shared_buffer import SharedReplayBuffer

        self.buffer = SharedReplayBuffer(self.all_args,
                                         self.num_agents,
                                         self.envs.observation_space[0] if self.envs else self.eval_envs.observation_space[0],
                                         share_observation_space,
                                         self.envs.action_space[0] if self.envs else self.eval_envs.action_space[0])

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    def restore(self):
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt', map_location=self.device)
        self.algo_module.actor.load_state_dict(policy_actor_state_dict)
