import os
import pickle
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1

import data
import nn_gp_layer
from envs import env_utils
from utils import nn_utils, misc_utils
from utils.logger_utils import EpochLogger

np.set_printoptions(precision=3)


class BrunoSAC:

    def __init__(self, train_envs, test_envs,
                 obs_dim, action_dim, reward_dim, env_params_dim, latent_dim, seq_len,
                 bruno_model, qf1, qf2, vf, policy,
                 policy_lr=1e-3, model_lr=1e-3, qf_lr=1e-3, alpha_lr=1e-3,
                 gamma=0.99, target_entropy='auto',
                 tau=0.005):

        # environment
        self.train_envs = train_envs
        self.train_envs_ids = env_utils.get_env_id(self.train_envs)
        self.test_envs = test_envs
        self.test_envs_ids = env_utils.get_env_id(self.test_envs)

        # dims
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.env_params_dim = env_params_dim

        self.latent_dim = latent_dim
        self.target_entropy = target_entropy if target_entropy != 'auto' else -np.prod(self.action_dim)
        print('target entropy', self.target_entropy)

        # logger
        self.logger = EpochLogger()

        # learning rates
        self.policy_lr = policy_lr
        self.model_lr = model_lr
        self.qf_lr = qf_lr
        self.vf_lr = qf_lr
        self.alpha_lr = alpha_lr

        # other params
        self.seq_len = seq_len
        self.gamma = gamma
        self.tau = tau

        # alpha
        log_alpha = tf.compat.v1.get_variable('log_alpha', dtype=tf.float32, initializer=0.)
        self.alpha = tf.exp(log_alpha)

        # placeholders
        self.iteration_var = tf1.placeholder(tf.int64, shape=None, name='iteration')
        self.obs_var = tf1.placeholder(tf.float32, shape=(None, self.obs_dim), name='obs')
        self.next_obs_var = tf1.placeholder(tf.float32, shape=(None, self.obs_dim), name='next_obs')
        self.actions_var = tf1.placeholder(tf.float32, shape=(None, self.action_dim), name='actions')
        self.rewards_var = tf1.placeholder(tf.float32, shape=(None, self.reward_dim), name='rewards')
        self.terminals_var = tf1.placeholder(tf.float32, shape=(None,), name='terminals')
        self.env_params_var = tf1.placeholder(tf.float32, shape=(None, self.env_params_dim), name='env_params')

        self.bruno_state = nn_gp_layer.State(tf1.placeholder(tf.float32, shape=(1, self.latent_dim)),
                                             tf1.placeholder(tf.float32, shape=(1, self.latent_dim)),
                                             tf1.placeholder(tf.float32, shape=()))

        # placeholders for sequences
        self.obs_seq_var = tf1.placeholder(tf.float32, shape=(None, self.seq_len, self.obs_dim), name='obs_seq_var')
        self.next_obs_seq_var = tf1.placeholder(tf.float32, shape=(None, self.seq_len, self.obs_dim),
                                                name='next_obs_seq_var')
        self.actions_seq_var = tf1.placeholder(tf.float32, shape=(None, self.seq_len, self.action_dim),
                                               name='actions_seq')
        self.rewards_seq_var = tf1.placeholder(tf.float32, shape=(None, self.seq_len, self.reward_dim),
                                               name='rewards_seq')
        self.terminals_seq_var = tf1.placeholder(tf.float32, shape=(None, self.seq_len), name='terminals_seq')
        self.env_params_seq_var = tf1.placeholder(tf.float32, shape=(None, self.seq_len, self.env_params_dim),
                                                  name='env_params_seq')

        # templates
        self.qf1 = tf1.make_template('qf1', qf1)
        self.qf2 = tf1.make_template('qf2', qf2)
        self.vf = tf1.make_template('vf_main', vf)
        self.vf_target = tf1.make_template('vf_target', vf)
        self.policy = tf1.make_template('policy', policy)
        self.bruno_model = bruno_model

        # outputs from the networks
        self.qf1_out = self.qf1(
            tf.concat([self.obs_seq_var, self.actions_seq_var, self.env_params_seq_var], axis=-1))
        qf2_out = self.qf2(tf.concat([self.obs_seq_var, self.actions_seq_var, self.env_params_seq_var], axis=-1))
        vf_out = self.vf(tf.concat([self.obs_seq_var, self.env_params_seq_var], axis=-1))
        vf_target_out = self.vf_target(tf.concat([self.next_obs_seq_var, self.env_params_seq_var], axis=-1))

        sampled_seq_actions, actions_seq_logprobs = self.bruno_model.sample_actions_sequence(policy=self.policy,
                                                                                             obs=self.obs_seq_var,
                                                                                             actions=self.actions_seq_var,
                                                                                             next_obs=self.next_obs_seq_var,
                                                                                             rewards=self.rewards_seq_var)
        self.bruno_states_seq = self.bruno_model.get_states_given_sequence(obs=self.obs_seq_var,
                                                                           actions=self.actions_seq_var,
                                                                           next_obs=self.next_obs_seq_var,
                                                                           rewards=self.rewards_seq_var)

        qf1_pi_out = self.qf1(tf.concat([self.obs_seq_var, sampled_seq_actions, self.env_params_seq_var], axis=-1))
        qf2_pi_out = self.qf2(tf.concat([self.obs_seq_var, sampled_seq_actions, self.env_params_seq_var], axis=-1))

        self.get_sampled_action = self.bruno_model.sample_action(policy=self.policy, gp_state=self.bruno_state,
                                                                 obs=self.obs_var)

        self.bruno_update_state = self.bruno_model.get_updated_state(gp_state=self.bruno_state, obs=self.obs_var,
                                                                     action=self.actions_var,
                                                                     next_obs=self.next_obs_var,
                                                                     reward=self.rewards_var)

        self.bruno_predictive_stats = self.bruno_model.decode_rewards_states_predictive_distribution(
            gp_state=self.bruno_state,
            obs=self.obs_var,
            action=self.actions_var)

        log_probs_model = bruno_model.get_sequence_model_likelihoods(obs=self.obs_seq_var,
                                                                     actions=self.actions_seq_var,
                                                                     next_obs=self.next_obs_seq_var,
                                                                     rewards=self.rewards_seq_var)
        # session and init weights
        self.sess = tf.Session()
        init_networks_params = tf.global_variables_initializer()
        self.sess.run(init_networks_params)
        self.saver = tf.train.Saver()

        print('number of parameters:', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

        # network parameters
        bruno_params = tf.trainable_variables(self.bruno_model.name)
        policy_params = tf.trainable_variables(self.policy.name)
        qf1_params = tf.trainable_variables(self.qf1.name)
        qf2_params = tf.trainable_variables(self.qf2.name)
        vf_params = tf.trainable_variables(self.vf.name)
        vf_target_params = tf.trainable_variables(self.vf_target.name)

        print('bruno model params', nn_utils.count_vars(self.bruno_model.name), bruno_params)
        print('policy params', nn_utils.count_vars(self.policy.name), policy_params)
        print('QF1', nn_utils.count_vars(self.qf1.name), qf1_params)
        print('QF2', nn_utils.count_vars(self.qf2.name), qf2_params)
        print('VF', nn_utils.count_vars(self.vf.name), vf_params)
        print('VF_target', nn_utils.count_vars(self.vf_target.name), vf_target_params)

        # losses
        self.q_target = tf.stop_gradient(tf.squeeze(self.rewards_seq_var) +
                                         (1. - self.terminals_seq_var) * self.gamma * vf_target_out)
        qf1_loss = 0.5 * tf.reduce_mean((self.q_target - self.qf1_out) ** 2)
        qf2_loss = 0.5 * tf.reduce_mean((self.q_target - qf2_out) ** 2)

        min_q_pi = tf.minimum(qf1_pi_out, qf2_pi_out)
        v_target = tf.stop_gradient(min_q_pi - self.alpha * actions_seq_logprobs)
        vf_loss = 0.5 * tf.reduce_mean((v_target - vf_out) ** 2)

        value_loss = qf1_loss + qf2_loss + vf_loss

        policy_loss = tf.reduce_mean(self.alpha * actions_seq_logprobs - min_q_pi)

        model_loss = - tf.reduce_mean(log_probs_model)

        alpha_loss = -tf.reduce_mean(log_alpha * tf.stop_gradient(actions_seq_logprobs + self.target_entropy))

        entropy = -tf.reduce_mean(actions_seq_logprobs)

        bruno_train_op = tf1.train.AdamOptimizer(learning_rate=self.model_lr).minimize(
            model_loss, var_list=bruno_params, name='bruno_opt')

        policy_train_op = tf1.train.AdamOptimizer(learning_rate=self.policy_lr).minimize(
            policy_loss, var_list=policy_params, name='policy_opt')

        with tf.control_dependencies([policy_train_op]):
            value_params = qf1_params + qf2_params + vf_params
            critics_train_op = tf1.train.AdamOptimizer(self.qf_lr).minimize(value_loss, var_list=value_params,
                                                                            name='qf_vf_opt')

        with tf.control_dependencies([critics_train_op]):
            alpha_train_op = tf1.train.AdamOptimizer(self.alpha_lr, name='alpha_opt').minimize(
                loss=alpha_loss, var_list=[log_alpha])
            target_update = tf.group([tf.assign(v_targ, (1. - self.tau) * v_targ + tau * v_main)
                                      for v_main, v_targ in zip(vf_params, vf_target_params)])

        self.actor_critic_train_step_ops = [policy_loss, model_loss, qf1_loss, qf2_loss, vf_loss, alpha_loss,
                                            self.qf1_out, qf2_out, vf_out, entropy,
                                            policy_train_op, critics_train_op, alpha_train_op, target_update]

        self.model_train_step_ops = [policy_loss, model_loss, qf1_loss, qf2_loss, vf_loss, alpha_loss,
                                     self.qf1_out, qf2_out, vf_out, entropy,
                                     bruno_train_op]

        # init the rest of variables
        target_init = tf.group(
            [tf.assign(v_targ, v_main) for v_main, v_targ in zip(vf_params, vf_target_params)])

        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                self.sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)

        init_new_vars_op = tf.initialize_variables(uninitialized_vars)
        self.sess.run(init_new_vars_op)
        self.sess.run(target_init)

    def reshape_env2bruno(self, x):
        x = np.squeeze(x)
        if len(x.shape) == 0:
            x = np.array([x])
        x = x[None, :]
        return x

    def reshape_bruno2env(self, x):
        if x.shape[-1] == 1:
            x = np.squeeze(x)[None]
        else:
            x = np.squeeze(x)
        return x

    def get_feed_dict(self, iteration, batch):

        feed_dict = {
            self.obs_seq_var: batch['observations'],
            self.actions_seq_var: batch['actions'],
            self.next_obs_seq_var: batch['next_observations'],
            self.rewards_seq_var: batch['rewards'],
            self.terminals_seq_var: batch['terminals'],
            self.env_params_seq_var: batch['env_params']
        }

        if iteration is not None:
            feed_dict[self.iteration_var] = iteration

        return feed_dict

    def do_actor_critic_training_steps(self, iteration, batch):
        feed_dict = self.get_feed_dict(iteration, batch)
        outs = self.sess.run(self.actor_critic_train_step_ops, feed_dict)
        self.logger.store(LossPi=outs[0], LossModel=outs[1], LossQ1=outs[2], LossQ2=outs[3],
                          LossV=outs[4], LossAlpha=outs[5], Q1Vals=outs[6], Q2Vals=outs[7], VVals=outs[8],
                          EntropyPi=outs[9], LossQ1Rel=outs[10])

    def do_model_training_steps(self, iteration, batch):
        feed_dict = self.get_feed_dict(iteration, batch)
        outs = self.sess.run(self.model_train_step_ops, feed_dict)
        self.logger.store(LossPi=outs[0], LossModel=outs[1], LossQ1=outs[2], LossQ2=outs[3],
                          LossV=outs[4], LossAlpha=outs[5], Q1Vals=outs[6], Q2Vals=outs[7], VVals=outs[8],
                          EntropyPi=outs[9], LossQ1Rel=outs[10])

    def get_action(self, current_bruno_state, observation, deterministic=False):
        feed_dict = {self.bruno_state: current_bruno_state, self.obs_var: observation}
        mu, sample = self.sess.run(self.get_sampled_action, feed_dict=feed_dict)
        return mu[0] if deterministic else sample[0]

    def get_updated_bruno_state(self, current_bruno_state, current_obs, action, next_obs, reward):
        feed_dict = {self.obs_var: current_obs, self.actions_var: action,
                     self.next_obs_var: next_obs, self.rewards_var: reward,
                     self.bruno_state: current_bruno_state}
        return self.sess.run(self.bruno_update_state, feed_dict=feed_dict)

    def get_bruno_predictive_stats(self, current_bruno_state, current_obs, action):
        return self.sess.run(self.bruno_predictive_stats, feed_dict={self.bruno_state: current_bruno_state,
                                                                     self.obs_var: current_obs,
                                                                     self.actions_var: action})

    def test(self, max_episode_length, train_iteration, n_episodes=1, plot_n_steps=0, plot_diagnostics=False,
             save_dir=None, n_reset_steps=None, dump_data=False):

        if save_dir is not None:
            ckpt_file = save_dir + 'params.ckpt'
            print('restoring parameters from', ckpt_file)
            self.saver.restore(self.sess, tf.train.latest_checkpoint(save_dir))
            gp_param_vals = self.sess.run([self.bruno_model.gp_layer.var, self.bruno_model.gp_layer.corr])
            print('GP var and corr', gp_param_vals)

        all_returns = []
        returns = defaultdict(list)
        rewards = defaultdict(list)

        for i, env in enumerate(self.test_envs):

            env.seed(i)

            rewards_env = {}
            env_id = env_utils.get_env_id(env)
            env_params = env.get_params()[None, :]

            for j in range(n_episodes):
                o = env.reset()

                if plot_n_steps > 0 and j == 0:
                    print('\n ---- ENVIRONMENT:', env_id)
                    print('initial state:', o)
                done_flag = False
                episode_history = []
                episode_rewards = []
                episode_length, episode_return = 0, 0.
                current_bruno_state = self.sess.run(self.bruno_model.prior)

                while episode_length < max_episode_length and not done_flag:

                    if n_reset_steps is not None and episode_length % n_reset_steps == 0:
                        print('%s return after %s steps :' % (env_id, episode_length), episode_return)
                        episode_return = 0.
                        o = env.reset()

                    current_obs = self.reshape_env2bruno(o)
                    a = self.get_action(current_bruno_state, current_obs, deterministic=True)
                    bruno_preds = self.get_bruno_predictive_stats(current_bruno_state, current_obs, a[None, :])
                    a = self.reshape_bruno2env(a)
                    o, r, done_flag, info = env.step(a)
                    episode_rewards.append(r)
                    episode_return += r
                    episode_length += 1
                    if (episode_length > max_episode_length - plot_n_steps or episode_length < plot_n_steps) and j == 0:
                        print(episode_length, 'action:', a, 'bruno state:', current_bruno_state.mu)
                        print('obs:', o)
                        env.render(mode='human')

                    next_obs = self.reshape_env2bruno(o)
                    r = self.reshape_env2bruno(r)
                    a = self.reshape_env2bruno(a)
                    done_flag = np.float32(np.array([done_flag]))
                    episode_history.append((current_obs, a, next_obs, r, done_flag, env_params, bruno_preds))

                    current_bruno_state = self.get_updated_bruno_state(current_bruno_state, current_obs, a,
                                                                       next_obs, r)
                if plot_n_steps and j == 0:
                    print('episode return:', episode_return)

                if plot_diagnostics and j == 0:
                    rews_seq = np.concatenate([x[3][:, None, :] for x in episode_history], axis=1)
                    nstate_seq = np.concatenate([x[2][:, None, :] for x in episode_history], axis=1)
                    preds_stats = [x[6] for x in episode_history]
                    misc_utils.plot_rewards_and_states(preds_stats, rews_seq, nstate_seq,
                                                       name='%s_%s_%s' % (train_iteration, env_id, episode_return))

                returns[env_id].append(episode_return)
                rewards_env[j] = episode_rewards
                all_returns.append(episode_return)
                kwargs = {'TestEpRet_' + env_id: episode_return, 'TestEpLen_' + env_id: episode_length}
                self.logger.store(**kwargs)

            rewards[env_id] = rewards_env
            print('test returns %s :' % env_id, returns[env_id])

        if dump_data:
            with open(save_dir + '/test_rewards.pkl', 'wb') as f:
                pickle.dump(rewards, f)

        print('average return', np.mean(all_returns))
        kwargs = {'TestAvgRet': np.mean(all_returns)}
        self.logger.store(**kwargs)
        return returns

    def train(self, max_episodes, n_exploration_episodes,
              min_collected_episodes,
              max_episode_length, max_test_episode_length,
              batch_size_episodes, batch_seq_len,
              n_save_iter, n_updates, replay_buffer, plot_n_steps=0,
              n_test_episodes=5, plot_diagnostics=False, save_dir=None, **kwargs):

        start_time = time.time()
        n_interactions = 0
        n_episodes = 0

        for iter_episodes in range(max_episodes):
            print(iter_episodes)

            for env in self.train_envs:
                # new episode
                env_id = env_utils.get_env_id(env)
                env_params = env.get_params()
                env_params = self.reshape_env2bruno(env_params)
                episode_history = data.Episode()
                episode_length, episode_return = 0, 0.

                current_obs = self.reshape_env2bruno(env.reset())
                current_bruno_state = self.sess.run(self.bruno_model.prior, feed_dict={})

                while episode_length < max_episode_length:
                    if iter_episodes > n_exploration_episodes:
                        a = self.get_action(current_bruno_state, current_obs)
                        a = self.reshape_bruno2env(a)
                    else:
                        a = env.action_space.sample()
                    next_obs, r, done_flag, _ = env.step(a)
                    episode_return += r
                    episode_length += 1

                    next_obs = self.reshape_env2bruno(next_obs)
                    r = self.reshape_env2bruno(r)
                    a = self.reshape_env2bruno(a)
                    done_flag = np.float32(np.array([done_flag]))

                    if not done_flag:
                        episode_history.append(current_obs, a, next_obs, r, done_flag, env_params)

                        current_bruno_state = self.get_updated_bruno_state(current_bruno_state, current_obs, a,
                                                                           next_obs, r)
                        current_obs = next_obs
                    else:
                        current_obs = self.reshape_env2bruno(env.reset())
                        current_bruno_state = self.sess.run(self.bruno_model.prior, feed_dict={})

                # save episode in the replay buffer
                replay_buffer.store(episode_history)

                n_interactions += episode_length
                n_episodes += 1
                kwargs = {'EpRet_' + env_id: episode_return, 'EpLen_' + env_id: episode_length}
                self.logger.store(**kwargs)

            if iter_episodes >= min_collected_episodes:
                # model updates
                for j in range(n_updates):
                    batch = replay_buffer.sample_batch(batch_size_episodes, batch_seq_len)
                    self.do_model_training_steps(iter_episodes, batch)

                # actor-critic updates
                if iter_episodes > n_exploration_episodes:
                    for j in range(n_updates):
                        batch = replay_buffer.sample_batch(batch_size_episodes, batch_seq_len)
                        self.do_actor_critic_training_steps(iter_episodes, batch)

                # collect some data
                if iter_episodes % 200 == 0 and plot_diagnostics:
                    b_s, obs_s, env_s = [], [], []
                    for k in range(1000):
                        batch = replay_buffer.sample_batch(batch_size_episodes, self.seq_len, shuffle=False)
                        b_s.append(self.sess.run(self.bruno_states_seq, feed_dict=self.get_feed_dict(0, batch)))
                        obs_s.append(batch['observations'])
                        env_s.append(batch['env_params'])
                    b_s = np.concatenate(b_s, axis=0)
                    obs_s = np.concatenate(obs_s, axis=0)
                    env_s = np.concatenate(env_s, axis=0)
                    np.savez('dump_%s' % iter_episodes, states=b_s, obs=obs_s, envs=env_s)
                    print('data dumped')

                if (iter_episodes + 1) % n_save_iter == 0:
                    # print params
                    print('current alpha', self.sess.run(self.alpha))
                    print('variance:\n', self.sess.run(self.bruno_model.gp_layer.var))
                    print('corr:\n', self.sess.run(self.bruno_model.gp_layer.corr))

                    # run test episodes
                    test_returns = self.test(n_episodes=n_test_episodes, max_episode_length=max_test_episode_length,
                                             plot_n_steps=plot_n_steps, plot_diagnostics=plot_diagnostics,
                                             train_iteration=iter_episodes)

                    # print stats
                    self.logger.log_tabular('Epoch', int(iter_episodes / n_save_iter))
                    self.logger.log_tabular('TotalEnvInteracts', n_interactions)
                    self.logger.log_tabular('TotalEpisodes', n_episodes)

                    for train_env_id in self.train_envs_ids:
                        self.logger.log_tabular('EpRet_' + train_env_id, average_only=True)

                    for test_env_id in self.test_envs_ids:
                        self.logger.log_tabular('TestEpRet_' + test_env_id, average_only=True)

                    self.logger.log_tabular('TestAvgRet', average_only=True)
                    self.logger.log_tabular('EntropyPi', with_min_and_max=True)
                    self.logger.log_tabular('Q1Vals', with_min_and_max=True)
                    self.logger.log_tabular('Q2Vals', with_min_and_max=True)
                    self.logger.log_tabular('VVals', with_min_and_max=True)
                    self.logger.log_tabular('LossAlpha', average_only=True)
                    self.logger.log_tabular('LossModel', average_only=True)
                    self.logger.log_tabular('LossPi', average_only=True)
                    self.logger.log_tabular('LossQ1', average_only=True)
                    self.logger.log_tabular('LossQ2', average_only=True)
                    self.logger.log_tabular('LossV', average_only=True)
                    self.logger.log_tabular('Time', time.time() - start_time)
                    self.logger.dump_tabular()

                    # save models
                    if save_dir is not None:
                        self.saver.save(self.sess, save_dir + '/params.ckpt')

                        if os.path.isfile(save_dir + '/meta.pkl'):
                            with open(save_dir + '/meta.pkl', 'rb') as f:
                                d = pickle.load(f)
                                d.update({n_interactions: test_returns})
                        else:
                            d = {n_interactions: test_returns}

                        with open(save_dir + '/meta.pkl', 'wb') as f:
                            pickle.dump(d, f)
