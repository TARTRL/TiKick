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
import sys
import os

from pathlib import Path

from tmarl.runners.base_evaluator import Evaluator
from tmarl.envs.football.football import RllibGFootball
from tmarl.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv


class FootballEvaluator(Evaluator):
    def __init__(self, argv):
        super(FootballEvaluator, self).__init__(argv)

    def setup_run_dir(self, all_args):
        dump_dir = Path(all_args.replay_save_dir)
        if not dump_dir.exists():
            os.makedirs(str(dump_dir))
        self.dump_dir = dump_dir

        return super(FootballEvaluator, self).setup_run_dir(all_args)

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


def main(argv):
    evaluator = FootballEvaluator(argv)
    evaluator.run()


if __name__ == "__main__":
    main(sys.argv[1:])
