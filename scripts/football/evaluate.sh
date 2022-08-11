#!/bin/bash

env="gfootball"

scenario="11_vs_11_kaggle"
num_agents=10

#scenario="academy_3_vs_1_with_keeper"
#num_agents=3

# scenario="academy_run_pass_and_shoot_with_keeper"
# num_agents=2

# scenario="academy_run_to_score_with_keeper"
# num_agents=1

algo="rmappo"

CUDA_VISIBLE_DEVICES=0 python3 ../../tmarl/runners/football/football_evaluator.py --env_name ${env} \
--algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} \
--n_eval_rollout_threads 2 --eval_num 1 --use_eval \
--replay_save_dir "../../results/$scenario/replay/" \
--model_dir "../../models/$scenario"
