import numpy as np

import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv

import tmarl.envs.football.env as football_env

class RllibGFootball(MultiAgentEnv):
    """An example of a wrapper for GFootball to make it compatible with rllib."""

    def __init__(self, all_args, rank, log_dir=None, isEval=False):
        
        self.num_agents = all_args.num_agents
        self.num_rollout = all_args.n_rollout_threads
        self.isEval = isEval
        self.rank = rank
        # create env
        # need_render = (rank == 0) and isEval
        need_render = (rank == 0)
        #    and (not isEval or self.use_behavior_cloning)
        self.env = football_env.create_environment(
            env_name=all_args.scenario_name, stacked=False,
            logdir=log_dir,
            representation=all_args.representation,
            rewards='scoring' if isEval else all_args.rewards,
            write_goal_dumps=False,
            write_full_episode_dumps=need_render,
            render=need_render,
            dump_frequency=1 if need_render else 0,
            number_of_left_players_agent_controls=self.num_agents,
            number_of_right_players_agent_controls=0,
            other_config_options={'action_set':'full'})
        # state
        self.last_loffside = np.zeros(11)
        self.last_roffside = np.zeros(11)
        # dimension
        self.action_size = 33

        if all_args.scenario_name == "11_vs_11_kaggle":
            self.avail_size = 20
        else:
            self.avail_size = 19

        if all_args.representation == 'raw':
            obs_space_dim = 268
            obs_space_low = np.zeros(obs_space_dim) - 1e6
            obs_space_high = np.zeros(obs_space_dim) + 1e6
            obs_space_type = 'float64'
        else:
            raise NotImplementedError

        self.action_space = [gym.spaces.Discrete(
            self.action_size) for _ in range(self.num_agents)]
        self.observation_space = [gym.spaces.Box(
            low=obs_space_low,
            high=obs_space_high,
            dtype=obs_space_type) for _ in range(self.num_agents)]
        self.share_observation_space = [gym.spaces.Box(
            low=obs_space_low,
            high=obs_space_high,
            dtype=obs_space_type) for _ in range(self.num_agents)]
            
    def reset(self):
        
        # available actions
        avail_actions = np.ones([self.num_agents, self.action_size])
        avail_actions[:, self.avail_size:] = 0
        # state
        self.last_loffside = np.zeros(11)
        self.last_roffside = np.zeros(11)
        # obs
        raw_obs = self.env.reset()
        raw_obs = self._notFullGame(raw_obs)
        obs = self.raw2vec(raw_obs)
        share_obs = obs.copy()

        return obs, share_obs, avail_actions

    def step(self, actions):
        # step
        actions = np.argmax(actions, axis=-1)
        raw_o, r, d, info = self.env.step(actions.astype('int32'))
        raw_o = self._notFullGame(raw_o)
        obs = self.raw2vec(raw_o)
        share_obs = obs.copy()
        # available actions
        avail_actions = np.ones([self.num_agents, self.action_size])
        avail_actions[:, self.avail_size:] = 0
        # translate to specific form
        rewards = []
        infos, dones = [], []
        for i in range(self.num_agents):
            infos.append(info)
            dones.append(d)
            reward = r[i] if self.num_agents > 1 else r
            reward = -0.01 if d and reward < 1 and not self.isEval else reward
            rewards.append(reward)
        rewards = np.expand_dims(np.array(rewards), axis=1)
        
        return obs, share_obs, rewards, dones, infos, avail_actions

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def close(self):
        self.env.close()

    def raw2vec(self, raw_obs):
        obs = []
        ally = np.array(raw_obs[0]['left_team'])
        ally_d = np.array(raw_obs[0]['left_team_direction'])
        enemy = np.array(raw_obs[0]['right_team'])
        enemy_d = np.array(raw_obs[0]['right_team_direction'])
        lo, ro = self.get_offside(raw_obs[0])
        for a in range(self.num_agents):
            # prepocess
            me = ally[int(raw_obs[a]['active'])]
            ball = raw_obs[a]['ball'][:2]
            ball_dist = np.linalg.norm(me - ball)
            enemy_dist = np.linalg.norm(me - enemy, axis=-1)
            to_enemy = enemy - me
            to_ally = ally - me
            to_ball = ball - me

            o = []
            # shape = 0
            o.extend(ally.flatten())
            o.extend(ally_d.flatten())
            o.extend(enemy.flatten())
            o.extend(enemy_d.flatten())
            # shape = 88
            o.extend(raw_obs[a]['ball'])
            o.extend(raw_obs[a]['ball_direction'])
            # shape = 94
            if raw_obs[a]['ball_owned_team'] == -1:
                o.extend([1, 0, 0])
            if raw_obs[a]['ball_owned_team'] == 0:
                o.extend([0, 1, 0])
            if raw_obs[a]['ball_owned_team'] == 1:
                o.extend([0, 0, 1])
            # shape = 97
            active = [0] * 11
            active[raw_obs[a]['active']] = 1
            o.extend(active)
            # shape = 108
            game_mode = [0] * 7
            game_mode[raw_obs[a]['game_mode']] = 1
            o.extend(game_mode)
            # shape = 115
            o.extend(raw_obs[a]['sticky_actions'][:10])
            # shape = 125)
            ball_dist = 1 if ball_dist > 1 else ball_dist
            o.extend([ball_dist])
            # shape = 126)
            o.extend(raw_obs[a]['left_team_tired_factor'])
            # shape = 137)
            o.extend(raw_obs[a]['left_team_yellow_card'])
            # shape = 148)
            o.extend(raw_obs[a]['left_team_active'])  # red cards
            # shape = 159)
            o.extend(lo)  # !
            # shape = 170)
            o.extend(ro)  # !
            # shape = 181)
            o.extend(enemy_dist)
            # shape = 192)
            to_ally[:, 0] /= 2
            o.extend(to_ally.flatten())
            # shape = 214)
            to_enemy[:, 0] /= 2
            o.extend(to_enemy.flatten())
            # shape = 236)
            to_ball[0] /= 2
            o.extend(to_ball.flatten())
            # shape = 238)

            steps_left = raw_obs[a]['steps_left']
            o.extend([1.0 * steps_left / 3001])       # steps left till end
            if steps_left > 1500:
                steps_left -= 1501                    # steps left till halfend
            steps_left = 1.0 * min(steps_left, 300.0)  # clip
            steps_left /= 300.0
            o.extend([steps_left])

            score_ratio = 1.0 * \
                (raw_obs[a]['score'][0] - raw_obs[a]['score'][1])
            score_ratio /= 5.0
            score_ratio = min(score_ratio, 1.0)
            score_ratio = max(-1.0, score_ratio)
            o.extend([score_ratio])
            # shape = 241
            o.extend([0.0] * 27)
            # shape = 268

            obs.append(o)
            
        return np.array(obs)

    def get_offside(self, obs):
        ball = np.array(obs['ball'][:2])
        ally = np.array(obs['left_team'])
        enemy = np.array(obs['right_team'])

        if obs['game_mode'] != 0:
            self.last_loffside = np.zeros(11, np.float32)
            self.last_roffside = np.zeros(11, np.float32)
            return np.zeros(11, np.float32), np.zeros(11, np.float32)

        need_recalc = False
        effective_ownball_team = -1
        effective_ownball_player = -1

        if obs['ball_owned_team'] > -1:
            effective_ownball_team = obs['ball_owned_team']
            effective_ownball_player = obs['ball_owned_player']
            need_recalc = True
        else:
            ally_dist = np.linalg.norm(ball - ally, axis=-1)
            enemy_dist = np.linalg.norm(ball - enemy, axis=-1)
            if np.min(ally_dist) < np.min(enemy_dist):
                if np.min(ally_dist) < 0.017:
                    need_recalc = True
                    effective_ownball_team = 0
                    effective_ownball_player = np.argmin(ally_dist)
            elif np.min(enemy_dist) < np.min(ally_dist):
                if np.min(enemy_dist) < 0.017:
                    need_recalc = True
                    effective_ownball_team = 1
                    effective_ownball_player = np.argmin(enemy_dist)

        if not need_recalc:
            return self.last_loffside, self.last_roffside

        left_offside = np.zeros(11, np.float32)
        right_offside = np.zeros(11, np.float32)

        if effective_ownball_team == 0:
            right_xs = [obs['right_team'][k][0] for k in range(1, 11)]
            right_xs = np.array(right_xs)
            right_xs.sort()

            for k in range(1, 11):
                if obs['left_team'][k][0] > right_xs[-1] and k != effective_ownball_player \
                   and obs['left_team'][k][0] > 0.0:
                    left_offside[k] = 1.0
        else:
            left_xs = [obs['left_team'][k][0] for k in range(1, 11)]
            left_xs = np.array(left_xs)
            left_xs.sort()

            for k in range(1, 11):
                if obs['right_team'][k][0] < left_xs[0] and k != effective_ownball_player \
                   and obs['right_team'][k][0] < 0.0:
                    right_offside[k] = 1.0

        self.last_loffside = left_offside
        self.last_roffside = right_offside

        return left_offside, right_offside


    def _notFullGame(self, raw_obs):
        # use this function when there are less than 11 players in the scenario
        left_ok = len(raw_obs[0]['left_team']) == 11
        right_ok = len(raw_obs[0]['right_team']) == 11
        if left_ok and right_ok:
            return raw_obs
        # set player's coordinate at (-1,0), set player's velocity as (0,0)
        for obs in raw_obs:
            obs['left_team'] = np.array(obs['left_team'])
            obs['right_team'] = np.array(obs['right_team'])
            obs['left_team_direction'] = np.array(obs['left_team_direction'])
            obs['right_team_direction'] = np.array(obs['right_team_direction'])
            while len(obs['left_team']) < 11:
                obs['left_team'] = np.concatenate([obs['left_team'], np.array([[-1,0]])], axis=0)
                obs['left_team_direction'] = np.concatenate([obs['left_team_direction'], np.zeros([1,2])], axis=0)
                obs['left_team_tired_factor'] = np.concatenate([obs['left_team_tired_factor'], np.zeros(1)], axis=0)
                obs['left_team_yellow_card'] = np.concatenate([obs['left_team_yellow_card'], np.zeros(1)], axis=0)
                obs['left_team_active'] = np.concatenate([obs['left_team_active'], np.ones(1)], axis=0)
            while len(obs['right_team']) < 11:
                obs['right_team'] = np.concatenate([obs['right_team'], np.array([[-1,0]])], axis=0)
                obs['right_team_direction'] = np.concatenate([obs['right_team_direction'], np.zeros([1,2])], axis=0)
        return raw_obs