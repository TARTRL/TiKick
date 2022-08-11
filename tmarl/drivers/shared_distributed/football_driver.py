from tqdm import tqdm
import numpy as np

from tmarl.drivers.shared_distributed.base_driver import Driver


def _t2n(x):
    return x.detach().cpu().numpy()


class FootballDriver(Driver):
    def __init__(self, config):
        super(FootballDriver, self).__init__(config)

    def run(self):
        self.trainer.prep_rollout()
        episodes = int(self.num_env_steps)
        total_num_steps = 0
        for episode in range(episodes):
            print('Episode {}:'.format(episode))

            self.eval(total_num_steps)
                
    def eval(self, total_num_steps):
        
        eval_episode_rewards = []
        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        agent_num = eval_obs.shape[1]
        used_buffer = self.buffer
        rnn_shape = [self.n_eval_rollout_threads, agent_num, *used_buffer.rnn_states_critic.shape[3:]]
        eval_rnn_states = np.zeros(rnn_shape, dtype=np.float32)
        eval_rnn_states_critic = np.zeros(rnn_shape, dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, agent_num, 1), dtype=np.float32)

        finished = None
        
        for eval_step in tqdm(range(3001)):
            self.trainer.prep_rollout()
            _, eval_action, eval_action_log_prob, eval_rnn_states, _ = \
                self.trainer.algo_module.get_actions(np.concatenate(eval_share_obs),
                                                     np.concatenate(eval_obs),
                                                     np.concatenate(eval_rnn_states),
                                                     None,
                                                     np.concatenate(eval_masks),
                                                     np.concatenate(eval_available_actions),
                                                     deterministic=True)
          
            eval_actions = np.array(
                np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(
                np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))


            if self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(
                    np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = \
                self.eval_envs.step(eval_actions_env)
            eval_rewards = eval_rewards.reshape([-1, agent_num])  # [roll_out, num_agents]

            if finished is None:
                eval_r = eval_rewards[:,:self.num_agents]
                eval_episode_rewards.append(eval_r)
                finished = eval_dones.copy()
            else:
                eval_r = (eval_rewards * ~finished)[:,:self.num_agents]
                eval_episode_rewards.append(eval_r)
                finished = eval_dones.copy() | finished

            eval_masks = np.ones(
                (self.n_eval_rollout_threads, agent_num, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), 1), dtype=np.float32)
            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)


            if finished.all() == True:
                break

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
