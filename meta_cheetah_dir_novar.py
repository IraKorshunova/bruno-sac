import argparse
import sys
import time

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1

from data import EnvironmentsReplayBuffer
from envs import env_utils
from nn_bruno_net import BrunoNet
from sac_bruno import BrunoSAC
from sac_oracle import OracleSAC
from utils import nn_utils, misc_utils
from utils.nn_utils import gaussian_loglikelihood, squashing_func

np.random.seed(317071)
tf1.reset_default_graph()
tf1.set_random_seed(0)


def value_nn(inputs, hidden_layers_sizes=(128,), scope_name='critic_nn'):
    input_shape = tf.shape(inputs)
    batch_size = input_shape[0]
    seq_len = input_shape[1]
    input_dim = nn_utils.int_shape(inputs)[-1]

    inputs = tf.reshape(inputs, (batch_size * seq_len, input_dim))

    with tf1.variable_scope(scope_name):
        for i, size in enumerate(hidden_layers_sizes):
            inputs = tf.layers.dense(inputs, size,
                                     activation=tf.nn.leaky_relu,
                                     name=scope_name + '_l' + str(i))
        out = tf.layers.dense(inputs, 1, activation=None, name=scope_name + '_l_out')

    out = tf.reshape(out, (batch_size, seq_len))
    return out


def policy_nn(inputs, hidden_layers_sizes=(128, 128,), scope_name='actor_nn'):
    with tf1.variable_scope(scope_name):
        for i, size in enumerate(hidden_layers_sizes):
            inputs = tf.layers.dense(inputs, size, activation=tf.nn.leaky_relu, name=scope_name + '_l%d' % i)

        mu = tf.layers.dense(inputs, action_dim, activation=None, name=scope_name + '_l_mu')

        LOG_STD_MAX = 2
        LOG_STD_MIN = -5
        log_std = tf.layers.dense(inputs, action_dim, activation=tf.tanh, name=scope_name + '_l_sigma')
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        sigma = tf.exp(log_std)
        samples = mu + sigma * tf.random.normal(tf.shape(mu))
        log_prob = gaussian_loglikelihood(samples, mu, sigma)
        mu, samples, log_prob = squashing_func(mu, samples, log_prob)
        return mu, samples, log_prob


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', dest='train', action='store_true')
    group.add_argument('--test', dest='train', action='store_false')
    parser.set_defaults(train=None)
    parser.add_argument('--oracle', action='store_true', help='use oracle (default: False)')
    args = parser.parse_args()

    print('train:', args.train)
    print('oracle:', args.oracle)

    config_name = __file__.split('/')[-1].replace('.py', '')
    config_name = config_name + '_oracle' if args.oracle else config_name

    train_params, test_params = [-1, 1.], [-1., 1.]
    train_envs_names, test_envs_names = env_utils.get_envs(env_type='HalfCheetahDirEnv',
                                                           train_env_params=train_params,
                                                           test_env_params=test_params)

    train_envs = [env_utils.initialize_envs([name], 43 + i)[0] for i, name in enumerate(train_envs_names)]
    test_envs = [env_utils.initialize_envs([name], 42)[0] for name in test_envs_names]

    print('Train envs:', env_utils.get_env_id(train_envs))
    print('Test envs:', env_utils.get_env_id(test_envs))

    action_dim = train_envs[0].action_space.shape[0]
    obs_dim = train_envs[0].observation_space.shape[0]

    print('observations dim', obs_dim)
    print('actions dim', action_dim)

    env_params_dim = len(train_params)  # 1-hot encoding
    extra_dims = 4  # extra latent dimensions
    seq_len = 100
    context_len = [25, 75]
    batch_size = 10

    bruno_model = BrunoNet(action_dim=action_dim, obs_dim=obs_dim, reward_dim=1, name='bruno',
                           maf_num_hidden=128,
                           n_maf_layers=6,
                           weight_norm=True,
                           extra_dims=extra_dims,
                           min_max_context_len=context_len,
                           use_posterior_var=False)

    replay_buffer = EnvironmentsReplayBuffer(env_params=train_params, obs_dim=obs_dim, act_dim=action_dim,
                                             size=int(1e6), encoding='1hot')

    if args.oracle:
        algorithm = OracleSAC(train_envs=train_envs, test_envs=test_envs,
                              replay_buffer=replay_buffer,
                              action_dim=action_dim, obs_dim=obs_dim, reward_dim=1, env_params_dim=env_params_dim,
                              qf1=value_nn, qf2=value_nn, vf=value_nn, policy=policy_nn,
                              seq_len=seq_len,
                              target_entropy='auto',
                              policy_lr=0.001, qf_lr=0.001, alpha_lr=0.001)

    else:
        algorithm = BrunoSAC(train_envs=train_envs, test_envs=test_envs,
                             action_dim=action_dim, obs_dim=obs_dim, reward_dim=1, env_params_dim=env_params_dim,
                             latent_dim=bruno_model.latent_ndim,
                             qf1=value_nn, qf2=value_nn, vf=value_nn, bruno_model=bruno_model, policy=policy_nn,
                             seq_len=seq_len,
                             target_entropy='auto',
                             policy_lr=0.001, qf_lr=0.001, alpha_lr=0.001, model_lr=0.001)

    if args.train:
        exp_id = '%s-%s' % (config_name, time.strftime("%Y_%m_%d", time.localtime()))
        save_dir = 'metadata/' + exp_id
        misc_utils.autodir(save_dir)
        print(exp_id)

        # output logs
        misc_utils.autodir('logs')
        sys.stdout = misc_utils.Logger('logs/%s.log' % exp_id)
        sys.stderr = sys.stdout

        algorithm.train(max_episodes=2500, n_exploration_episodes=250,
                        min_collected_episodes=20,
                        max_episode_length=200, max_test_episode_length=200,
                        n_updates=200,
                        batch_size_episodes=batch_size, batch_seq_len=seq_len,
                        replay_buffer=replay_buffer,
                        n_save_iter=20, plot_n_steps=0, plot_diagnostics=False, n_test_episodes=10,
                        save_dir=save_dir)
    else:
        save_dir = misc_utils.find_model_metadata('metadata', config_name)
        algorithm.test(max_episode_length=1000, train_iteration=0, n_episodes=10, save_dir=save_dir,
                       plot_n_steps=1000, dump_data=True, plot_diagnostics=True)
