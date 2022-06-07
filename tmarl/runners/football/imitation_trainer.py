#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 The TARTRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""""""
from sre_constants import OP_IGNORE
import sys
import os
import torch
import numpy as np

from collections import deque
from pathlib import Path
from tqdm import tqdm

from tmarl.runners.base_evaluator import Evaluator
from tmarl.envs.football.football import RllibGFootball
from tmarl.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from tmarl.networks.policy_network import PolicyNetwork

def _t2n(x):
    return x.detach().cpu().numpy()

class ImitationTrainer(Evaluator):
    def __init__(self, argv):
        super(ImitationTrainer, self).__init__(argv)
        self.device = self.config['device']
        self.num_agents = self.config['num_agents']
        self.run_dir = self.config['run_dir']
    def setup_run_dir(self, all_args):
        dump_dir = Path(all_args.replay_save_dir)
        if not dump_dir.exists():
            os.makedirs(str(dump_dir))
        self.dump_dir = dump_dir

        return super(ImitationTrainer, self).setup_run_dir(all_args)

    def make_eval_env(self, all_args, Env_Class, SubprocVecEnv, DummyVecEnv):
        
        def get_env_fn(rank):
            def init_env():
                env = Env_Class(all_args, rank, log_dir=str(self.dump_dir), isEval=True)
                env.seed(all_args.seed * 50000 + rank * 10000)
                return env
            return init_env

        if all_args.n_eval_rollout_threads == 1:
            return DummyVecEnv([get_env_fn(0)])
        else:
            return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

    def extra_args_func(self, args, parser):
        parser.add_argument('--scenario_name', type=str,
                            default='simple_spread', help="Which scenario to run on")
        parser.add_argument('--num_agents', type=int,
                            default=0, help="number of players")

        # football config
        parser.add_argument('--representation', type=str,
                            default='raw', help="format of the observation in gfootball env")
        parser.add_argument('--rewards', type=str,
                            default='scoring', help="format of the reward in gfootball env")
        parser.add_argument("--render_only", action='store_true', default=False,
                            help="if ture, render without training")

        all_args = parser.parse_known_args(args)[0]
        return all_args

    def get_env(self):
        return RllibGFootball, ShareSubprocVecEnv, ShareDummyVecEnv

    def init_driver(self):
        if not self.all_args.separate_policy:
            from tmarl.drivers.shared_distributed.football_driver import FootballDriver as Driver
        else:
            raise NotImplementedError
        driver = Driver(self.config)
        return driver

    def gen_buffer(self, num_episodes, period_len):
        obs_buffer = []
        action_buffer = []
        for episode in range(num_episodes):
            eval_episode_rewards = []
            eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
            agent_num = eval_obs.shape[1]
            rnn_shape = [self.all_args.n_eval_rollout_threads,agent_num,self.all_args.recurrent_N,self.all_args.hidden_size]
            eval_rnn_states = np.zeros(rnn_shape, dtype=np.float32)
            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, agent_num, 1), dtype=np.float32)
            finished = None
            # obs_episode = []
            # action_episode = []
            replay_obs = deque(maxlen=period_len)
            replay_action = deque(maxlen=period_len)
            for eval_step in range(3001):
                self.teacher.eval()
                eval_action, eval_action_log_prob, eval_rnn_states = \
                self.teacher(np.concatenate(eval_obs),
                                np.concatenate(eval_rnn_states),
                                np.concatenate(eval_masks),
                                np.concatenate(eval_available_actions),
                                deterministic=True
                                )
                eval_actions = np.array(
                    np.split(_t2n(eval_action), self.all_args.n_eval_rollout_threads))
                eval_rnn_states = np.array(
                    np.split(_t2n(eval_rnn_states), self.all_args.n_eval_rollout_threads))

                if self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                    eval_actions_env = np.squeeze(
                        np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
                else:
                    raise NotImplementedError
                replay_obs.append(eval_obs)
                replay_action.append(eval_actions_env)
                if len(replay_obs) == period_len:
                    # print("Current Step:", eval_step)
                    # obs_episode.append(eval_obs)
                    # action_episode.append(eval_actions_env)
                    obs_buffer.append(eval_obs)
                    action_buffer.append(eval_actions_env)

                eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = \
                self.eval_envs.step(eval_actions_env)
                eval_rewards = eval_rewards.reshape([-1, agent_num]) 

                if finished is None:
                    eval_r = eval_rewards[:,:self.num_agents]
                    eval_episode_rewards.append(eval_r)
                    finished = eval_dones.copy()
                else:
                    eval_r = (eval_rewards * ~finished)[:,:self.num_agents]
                    eval_episode_rewards.append(eval_r)
                    finished = eval_dones.copy() | finished

                eval_masks = np.ones(
                    (self.all_args.n_eval_rollout_threads, agent_num, 1), dtype=np.float32)
                eval_masks[eval_dones == True] = np.zeros(
                    ((eval_dones == True).sum(), 1), dtype=np.float32)
                eval_rnn_states[eval_dones == True] = np.zeros(
                    ((eval_dones == True).sum(), self.all_args.recurrent_N, self.all_args.hidden_size), dtype=np.float32)


                if finished.all() == True:
                    break

            # obs_buffer.append(obs_episode)
            # action_buffer.append(action_episode)

            eval_episode_rewards = np.array(eval_episode_rewards)  # [step,rollout,num_agents]

            ally_goal = np.sum((eval_episode_rewards == 1), axis=0)
            enemy_goal = np.sum((eval_episode_rewards == -1), axis=0)
            net_goal = np.sum(eval_episode_rewards, axis=0)
            winning_rate = np.mean(net_goal, axis=-1)
            eval_env_infos = {}
            eval_env_infos['eval_average_winning_rate'] = winning_rate>0
            eval_env_infos['eval_average_losing_rate'] = winning_rate<0
            eval_env_infos['eval_average_draw_rate'] = winning_rate==0
            eval_env_infos['eval_average_ally_score'] = ally_goal
            eval_env_infos['eval_average_enemy_score'] = enemy_goal
            eval_env_infos['eval_average_net_score'] = net_goal
            print("\tSuccess Rate: " + str(np.mean(winning_rate>0)) )
        return obs_buffer, action_buffer, eval_available_actions

    def imitate(self):
        # print(self.all_args)
        period_len = 10
        self.teacher = PolicyNetwork(self.all_args,
                          self.eval_envs.observation_space[0],
                        #   self.config['eval_envs'].share_observation_space[0],
                          self.eval_envs.action_space[0],
                          device=self.device,
                        #   output_logit=True
                          )
        # state_dict = torch.load('./scripts/football/imitated_academy_3_vs_1_with_keeper.pt', map_location=self.config['device'])
        state_dict = torch.load((str(self.all_args.model_dir))+'/actor.pt', map_location=self.config['device'])
        self.teacher.load_state_dict(state_dict)
        self.student = PolicyNetwork(self.all_args,
                          self.eval_envs.observation_space[0],
                        #   self.config['eval_envs'].share_observation_space[0],
                          self.eval_envs.action_space[0],
                          device=self.device,
                          output_logit=True
                          )
        print("Generating Replay Buffer...")
        # print(self.all_args.epochs)
        obs_buffer, action_buffer, eval_available_actions = self.gen_buffer(self.all_args.eval_num, period_len)
        # print(len(obs_buffer))
        print('Training Networks...')
        obs_buffer = np.array(obs_buffer)
        action_buffer = np.array(action_buffer)
        # print(obs_buffer.shape)
        # print(action_buffer.shape)
        # obs_buffer = obs_buffer.reshape(-1, obs_buffer.shape[2],obs_buffer.shape[3],obs_buffer.shape[4])
        # action_buffer = action_buffer.reshape(-1, action_buffer.shape[2], action_buffer.shape[3],action_buffer.shape[4])
        # print(obs_buffer.shape)
        optimizer = torch.optim.Adam(self.student.parameters(),lr=self.all_args.lr,eps=self.all_args.opti_eps)
        loss_f = torch.nn.CrossEntropyLoss()
        pbar = tqdm(range(self.all_args.epochs))
        for epoch in pbar:
            idx = np.arange(len(obs_buffer))
            np.random.shuffle(idx)
            obs_buffer = obs_buffer[idx]
            action_buffer = action_buffer[idx]
            dataset = zip(obs_buffer,action_buffer)
            # dataset = dataset[idx]
            # obs_buffer = obs_buffer[idx]
            # action_buffer = action_buffer[idx]
            losses = []
            for obs, label in dataset:
                # print(obs.shape)
                agent_num = obs.shape[1]
                # print("agent num:", agent_num)
                rnn_shape = [self.all_args.n_eval_rollout_threads,agent_num,self.all_args.recurrent_N,self.all_args.hidden_size]
                rnn_states = np.zeros(rnn_shape, dtype=np.float32)
                masks = np.ones((self.all_args.n_eval_rollout_threads, agent_num, 1), dtype=np.float32)
                action_logits, rnn_states = self.student(np.concatenate(obs),
                                np.concatenate(rnn_states),
                                np.concatenate(masks),
                                np.concatenate(eval_available_actions),
                                deterministic=True,
                                output_logit=True
                                )
                # print(label.shape)
                label = torch.tensor(label.reshape(-1, action_logits.shape[1])).to(self.device)
                optimizer.zero_grad()
                loss = loss_f(action_logits, label)
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().cpu().numpy())
                pbar.set_description("Loss: %f" % (np.array(losses).mean()))
            
            torch.save(self.student.state_dict(), 'imitated_'+self.all_args.scenario_name +'.pt')
                # print(action_logits.shape, label.shape)







def main(argv):
    evaluator = ImitationTrainer(argv)
    # print(argv)
    evaluator.imitate()


if __name__ == "__main__":
    main(sys.argv[1:])
