#!/bin/bash

env="gfootball"

scenario="academy_3_vs_1_with_keeper"
num_agents=3

algo="rmappo"
exp="eval_3v2"

CUDA_VISIBLE_DEVICES=0 python3 ../../tmarl/runners/football/football_evaluator.py --env_name ${env} \
--algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} \
--n_eval_rollout_threads 32 --eval_num 2 --use_eval \
--replay_save_dir "../../results/academy_3_vs_1_with_keeper/replay/" \
--model_dir "../../models/academy_3_vs_1_with_keeper"
