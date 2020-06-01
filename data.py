import numpy as np


class EnvironmentsReplayBuffer:

    def __init__(self, env_params, obs_dim, act_dim, size, encoding='1hot'):

        self.n_env = len(env_params)
        self.env_idxs = list(range(self.n_env))
        print('n envs', self.n_env)
        self.buffers = []
        for i in range(self.n_env):
            self.buffers.append(ExperiencesReplayBuffer(obs_dim, act_dim, int(size / self.n_env)))

        self.env_params2env_idx = dict(zip(env_params, self.env_idxs))
        self.env_idx2env_params = dict((v, k) for k, v in self.env_params2env_idx.items())
        print(self.env_params2env_idx)
        print(self.env_idx2env_params)
        self.I = np.eye(self.n_env)
        self.encoding = encoding

    def encode_env_idx(self, env_idx):
        if self.encoding == '1hot':
            return self.I[env_idx]
        else:
            return self.env_idx2env_params[env_idx]

    def encode_env_param(self, env_params):
        out = env_params[0][0] if len(env_params[0]) == 1 else tuple(np.squeeze(env_params[0]))

        if self.encoding is None:
            if type(out) is not np.ndarray:
                out = np.asarray(out)
                if out.shape == ():
                    out = out[None]
            return out
        else:
            env_idx = self.env_params2env_idx[out]
            return self.encode_env_idx(env_idx)

    def store(self, episode):

        observations, actions, next_observations, rewards, terminals, env_params = episode.get_episode_data()
        p = env_params[0][0] if len(env_params[0]) == 1 else tuple(np.squeeze(env_params[0]))
        env_idx = self.env_params2env_idx[p]
        self.buffers[env_idx].bulk_store(obs=observations, act=actions, rew=rewards, next_obs=next_observations,
                                         done=terminals)

    def sample_batch(self, batch_size, seq_len, shuffle=None):
        """
        returns a dict, where each entry is batch_size_episodes x batch_size_steps x n_dims
        """
        observations = []
        next_observations = []
        actions = []
        rewards = []
        terminals = []
        env_params = []

        for i in np.random.choice(self.env_idxs, size=batch_size):
            b = self.buffers[i].sample_batch(batch_size=seq_len)
            observations.append(b['observations'])
            next_observations.append(b['next_observations'])
            actions.append(b['actions'])
            rewards.append(b['rewards'])
            terminals.append(b['terminals'])
            env_params.append([self.encode_env_idx(i)] * seq_len)

        observations = np.stack(observations, axis=0)
        next_observations = np.stack(next_observations, axis=0)
        actions = np.stack(actions, axis=0)
        rewards = np.stack(rewards, axis=0)
        terminals = np.stack(terminals, axis=0)
        env_params = np.stack(env_params, axis=0)
        if len(env_params.shape) == 2:
            env_params = env_params[:, :, None]

        return {'observations': observations,
                'next_observations': next_observations,
                'actions': actions,
                'rewards': rewards,
                'terminals': terminals,
                'env_params': env_params}


class ExperiencesReplayBuffer:

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size, 1], dtype=np.float32)
        self.terminals_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.terminals_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def bulk_store(self, obs, act, rew, next_obs, done):
        n = len(obs)
        if self.ptr + n < self.max_size:
            self.obs1_buf[self.ptr:self.ptr + n] = obs
            self.obs2_buf[self.ptr:self.ptr + n] = next_obs
            self.acts_buf[self.ptr:self.ptr + n] = act
            self.rews_buf[self.ptr:self.ptr + n] = rew
            self.terminals_buf[self.ptr:self.ptr + n] = done
            self.ptr = (self.ptr + n) % self.max_size
            self.size = min(self.size + n, self.max_size)
        else:
            for i in range(n):
                self.store(obs[i], act[i], rew[i], next_obs[i], done[i])

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)

        return {'observations': self.obs1_buf[idxs],
                'next_observations': self.obs2_buf[idxs],
                'actions': self.acts_buf[idxs],
                'rewards': self.rews_buf[idxs],
                'terminals': self.terminals_buf[idxs]}


class Episode:

    def __init__(self):
        self.observations = []
        self.actions = []
        self.next_observations = []
        self.rewards = []
        self.terminals = []
        self.env_params = []

    def append(self, obs, a, next_obs, r, t, env):
        self.observations.append(obs)
        self.actions.append(a)
        self.next_observations.append(next_obs)
        self.rewards.append(r)
        self.terminals.append(t)
        self.env_params.append(env)

    def get_episode_data(self):
        observations = np.concatenate(self.observations, axis=0)
        actions = np.concatenate(self.actions, axis=0)
        next_observations = np.concatenate(self.next_observations, axis=0)
        rewards = np.concatenate(self.rewards, axis=0)
        terminals = np.concatenate(self.terminals, axis=0)
        env_params = np.concatenate(self.env_params, axis=0)

        return observations, actions, next_observations, rewards, terminals, env_params
