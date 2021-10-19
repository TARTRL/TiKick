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

import argparse

def get_config():

    parser = argparse.ArgumentParser(
        description='TiKick', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str,
                        default='rmappo', choices=["rmappo"])

    parser.add_argument("--experiment_name", type=str, default="check",
                        help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for numpy/torch")
    parser.add_argument("--disable_cuda", action='store_true', default=False,
                        help="by default False, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True,
                        help="by default, make sure random seed effective. if set, bypass such function.")

    parser.add_argument("--n_rollout_threads", type=int, default=2,
                        help="Number of parallel envs for training rollout")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for evaluating rollout")
    parser.add_argument("--n_render_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for rendering rollout")
    parser.add_argument("--eval_num", type=int, default=1,
                        help='Number of environment steps to evaluate (default: 1)')

    # env parameters
    parser.add_argument("--env_name", type=str, default='StarCraft2',
                        help="specify the name of environment")
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int,
                        default=200, help="Max length for any episode")

    # network parameters
    parser.add_argument("--separate_policy", action='store_true',
                        default=False, help='Whether agent seperate the policy')
    parser.add_argument("--use_centralized_V", action='store_false',
                        default=True, help="Whether to use centralized V function")
    parser.add_argument("--use_conv1d", action='store_true',
                        default=False, help="Whether to use conv1d")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", action='store_true',
                        default=False, help="Whether to use stacked_frames")
    parser.add_argument("--hidden_size", type=int, default=256,
                        help="Dimension of hidden layers for actor/critic networks")  # TODO @zoeyuchao. The same comment might in need of change.
    parser.add_argument("--layer_N", type=int, default=3,
                        help="Number of layers for actor/critic networks")
    parser.add_argument("--activation_id", type=int,
                        default=1, help="choose 0 to use tanh, 1 to use relu, 2 to use leaky relu, 3 to use elu")
    parser.add_argument("--use_popart", action='store_true', default=False,
                        help="by default False, use PopArt to normalize rewards.")
    parser.add_argument("--use_valuenorm", action='store_false', default=True,
                        help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization", action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")
    parser.add_argument("--cnn_layers_params", type=str, default=None,
                        help="The parameters of cnn layer")
    parser.add_argument("--use_maxpool2d", action='store_true',
                        default=False, help="Whether to apply layernorm to the inputs")

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", action='store_true',
                        default=False, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", action='store_false',
                        default=True, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1,
                        help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=25,
                        help="Time length of chunks used to train a recurrent_policy")
    parser.add_argument("--use_influence_policy", action='store_true',
                        default=False, help='use a recurrent policy')
    parser.add_argument("--influence_layer_N", type=int, default=1,
                        help="Number of layers for actor/critic networks")


    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--tau", type=float, default=0.995,
                        help='soft update polyak (default: 0.995)')
    parser.add_argument("--critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_policy_vhead",
                        action='store_true', default=False,
                        help="by default, do not use policy vhead. if set, use policy vhead.")
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True,
                        help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')                   
    parser.add_argument("--policy_value_loss_coef", type=float,
                        default=1, help='policy value loss coefficient (default: 0.5)')
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True,
                        help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True,
                        help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks",
                        action='store_false', default=True,
                        help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",
                        action='store_false', default=True,
                        help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float,
                        default=10.0, help=" coefficience of huber loss.")

    # save parameters
    parser.add_argument("--save_interval", type=int, default=1,
                        help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=5,
                        help="time duration between contiunous twice log printing.")

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', default=False,
                        help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=25,
                        help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=64,
                        help="number of episodes of a single evaluation.")

    # pretrained parameters
    parser.add_argument("--model_dir", type=str, default=None,
                        help="by default None. set the path to pretrained model.")

    parser.add_argument("--replay_save_dir", type=str, default=None,
                        help="replay file save dir")

    # replay buffer parameters



    return parser
