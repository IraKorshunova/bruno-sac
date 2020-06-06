import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1

from nn_bijective_layers import MAF
from nn_gp_layer import GaussianRecurrentLayer
from utils import nn_utils


class BrunoNet(object):
    def __init__(self, action_dim, obs_dim, reward_dim, name,
                 min_max_context_len,
                 maf_num_hidden=32, n_maf_layers=2,
                 weight_norm=True, debug_mode=False, extra_dims=0, corr_init=0.1,
                 learn_covariance=True, learn_variance=True, use_posterior_var=False, model_next_state=False):
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.reward_dim = reward_dim
        self.name = name
        self.extra_dims = extra_dims
        self.min_max_context_len = min_max_context_len
        self.use_posterior_var = use_posterior_var
        self.model_next_state = model_next_state

        self.noise_entropy = 0.5 * extra_dims * np.log(2 * np.pi * np.e)
        print('noise entropy', self.noise_entropy)  # substract this from the loss -> lower bound on the marginal LL

        self.latent_ndim = reward_dim + obs_dim + extra_dims if model_next_state else reward_dim + extra_dims

        with tf1.variable_scope(name):
            self.maf = MAF(input_size=self.latent_ndim, name=name + '/model/maf_next_obs_reward',
                           n_maf_layers=n_maf_layers,
                           n_units=maf_num_hidden, weight_norm=weight_norm, debug_mode=debug_mode)

            self.gp_layer = GaussianRecurrentLayer(ndim=self.latent_ndim, corr_init=corr_init,
                                                   learn_covariance=learn_covariance,
                                                   learn_variance=learn_variance, name='model/gp')
            self.prior = self.gp_layer.prior

    def reset(self):
        return self.gp_layer.reset()

    def set_state(self, state):
        self.gp_layer.set_state(state)

    def encode(self, obs, actions, next_obs, rewards):

        input_shape = tf.shape(obs)
        batch_size = input_shape[0]
        seq_len = input_shape[1] if input_shape.shape == 3 else 1

        obs = tf.reshape(obs, (batch_size * seq_len, self.obs_dim))
        next_obs = tf.reshape(next_obs, (batch_size * seq_len, self.obs_dim))
        actions = tf.reshape(actions, (batch_size * seq_len, self.action_dim))
        rewards = tf.reshape(rewards, (batch_size * seq_len, self.reward_dim))

        condition = tf.concat([obs, actions], axis=-1)
        input = tf.concat([rewards, next_obs], axis=-1) if self.model_next_state else rewards
        if self.extra_dims > 0:
            noise = tf.random_normal(shape=(batch_size * seq_len, self.extra_dims))
            input = tf.concat([input, noise], axis=-1)

        jacob = tf.zeros(batch_size * seq_len)

        z, jacob = self.maf.forward_and_jacobian(input, jacob, condition=condition)

        # reshape sequences into their original shape (batch_size, seq_len, input_dim)
        if input_shape.shape == 3:
            z = tf.reshape(z, shape=(batch_size, seq_len, self.latent_ndim))
            jacob = tf.reshape(jacob, (batch_size, seq_len))

        return z, jacob

    def decode(self, z, obs, actions):

        input_shape = tf.shape(z)
        batch_size = input_shape[0]
        seq_len = input_shape[1] if input_shape.shape == 3 else 1

        obs = tf.reshape(obs, (batch_size * seq_len, self.obs_dim))
        actions = tf.reshape(actions, (batch_size * seq_len, self.action_dim))
        z = tf.reshape(z, (batch_size * seq_len, self.latent_ndim))

        condition = tf.concat([obs, actions], axis=-1)
        bwd_jacob = tf.zeros(batch_size * seq_len)

        output, neg_jacob = self.maf.backward(z, bwd_jacob, condition=condition)
        # the backward jacobian may not be safe to use since I don't remember testing it

        if self.extra_dims > 0:
            output = output[:, :-self.extra_dims]

        # reshape sequences into their original shape (batch_size, seq_len, input_dim)
        if input_shape.shape == 3:
            output = tf.reshape(output, shape=(batch_size, seq_len, self.latent_ndim - self.extra_dims))
            neg_jacob = tf.reshape(neg_jacob, (batch_size, seq_len))

        return output, neg_jacob

    def get_sequence_model_likelihoods(self, obs, actions, next_obs, rewards):

        z, jacob = self.encode(obs=obs, actions=actions, next_obs=next_obs,
                               rewards=rewards)

        context_len = tf.random.uniform(shape=[], minval=self.min_max_context_len[0],
                                        maxval=self.min_max_context_len[1], dtype=tf.int32)
        self.gp_layer.reset()
        self.gp_layer.bulk_update_distribution(z[:, :context_len, :])

        llp_model = self.gp_layer.get_sequence_log_likelihood(z)

        log_probs_model = llp_model + jacob
        return log_probs_model

    def get_states_given_sequence(self, obs, actions, next_obs, rewards):
        batch_size = tf.shape(obs)[0]
        seq_len = nn_utils.int_shape(obs)[1]

        z, jacob = self.encode(obs, actions, next_obs, rewards)

        bruno_states_seq = []

        self.gp_layer.reset()
        self.gp_layer.replicate_state(batch_size)

        for i in range(seq_len):
            mu, var = self.gp_layer.get_posterior_params()
            bruno_states_seq.append(mu)
            self.gp_layer.update_distribution(z[:, i, :])

        bruno_states_seq = tf.stack(bruno_states_seq, axis=1)
        return bruno_states_seq

    def get_updated_state(self, gp_state, obs, action, next_obs, reward):
        with tf1.variable_scope(tf1.get_variable_scope(), reuse=True):
            z, _ = self.encode(obs=obs, actions=action, next_obs=next_obs, rewards=reward)
            return self.gp_layer.get_updated_state(gp_state, z)

    def sample_actions_sequence(self, policy, obs, actions, next_obs, rewards):
        batch_size = tf.shape(obs)[0]
        seq_len = nn_utils.int_shape(obs)[1]

        z, jacob = self.encode(obs, actions, next_obs, rewards)

        act_samples = []
        act_probs = []

        self.gp_layer.reset()
        self.gp_layer.replicate_state(batch_size)

        for i in range(seq_len):
            posterior_mu, posterior_var = self.gp_layer.get_posterior_params()
            if self.use_posterior_var:
                policy_inputs = tf.concat([obs[:, i, :], posterior_mu, posterior_var], axis=-1)
            else:
                policy_inputs = tf.concat([obs[:, i, :], posterior_mu], axis=-1)

            _, act_sample, act_prob = policy(inputs=policy_inputs)
            act_samples.append(act_sample)
            act_probs.append(act_prob)

            self.gp_layer.update_distribution(z[:, i, :])

        act_samples = tf.stack(act_samples, axis=1)
        act_probs = tf.stack(act_probs, axis=1)

        return act_samples, act_probs

    def sample_action(self, policy, gp_state, obs):
        posterior_mu, posterior_var = self.gp_layer.get_posterior_params_given_state(gp_state)
        if self.use_posterior_var:
            policy_inputs = tf.concat([obs, posterior_mu, posterior_var], axis=-1)
        else:
            policy_inputs = tf.concat([obs, posterior_mu], axis=-1)

        act_mean, act_sample, _ = policy(inputs=policy_inputs)
        return act_mean, act_sample

    def decode_rewards_states_predictive_distribution(self, gp_state, obs, action):
        mu, var, _ = gp_state
        median, _ = self.decode(mu, obs, action)
        low_percentile, _ = self.decode(mu - 2. * tf.sqrt(var), obs, action)
        high_percentile, _ = self.decode(mu + 2. * tf.sqrt(var), obs, action)
        return median
