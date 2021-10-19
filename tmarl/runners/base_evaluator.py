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


import random

import numpy as np
import torch

from tmarl.configs.config import get_config
from tmarl.runners.base_runner import Runner

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Evaluator(Runner):
    def __init__(self, argv,program_type=None, client=None):
        super().__init__(argv)

        parser = get_config()
        all_args = self.extra_args_func(argv, parser)

        all_args.cuda = not all_args.disable_cuda

        self.algorithm_name = all_args.algorithm_name

        # cuda
        if not all_args.disable_cuda and torch.cuda.is_available():
            device = torch.device("cuda:0")

            if all_args.cuda_deterministic:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
        else:
            print("choose to use cpu...")
            device = torch.device("cpu")


        # run dir
        run_dir = self.setup_run_dir(all_args)

        # env init
        Env_Class, SubprocVecEnv, DummyVecEnv = self.get_env()
        eval_envs = self.env_init(
            all_args, Env_Class, SubprocVecEnv, DummyVecEnv)
        num_agents = all_args.num_agents

        config = {
            "all_args": all_args,
            "envs": None,
            "eval_envs": eval_envs,
            "num_agents": num_agents,
            "device": device,
            "run_dir": run_dir,
        }
        self.all_args, self.envs, self.eval_envs, self.config \
            = all_args, None, eval_envs, config
        self.driver = self.init_driver()
        
    def run(self):
        # run experiments
        self.driver.run()
        self.stop()

    def stop(self):
        pass

    def extra_args_func(self, argv, parser):
        raise NotImplementedError

    def get_env(self):
        raise NotImplementedError

    def init_driver(self):
        raise NotImplementedError

    def make_eval_env(self, all_args, Env_Class, SubprocVecEnv, DummyVecEnv):
        def get_env_fn(rank):
            def init_env():
                env = Env_Class(all_args)
                env.seed(all_args.seed * 50000 + rank * 10000)
                return env

            return init_env

        if all_args.n_eval_rollout_threads == 1:
            return DummyVecEnv([get_env_fn(0)])
        else:
            return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

    def env_init(self, all_args, Env_Class, SubprocVecEnv, DummyVecEnv):
        eval_envs = self.make_eval_env(
            all_args, Env_Class, SubprocVecEnv, DummyVecEnv) if all_args.use_eval else None
        return eval_envs

    def setup_run_dir(self, all_args):
        return None
