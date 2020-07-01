import collections

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1

State = collections.namedtuple('State', ['mu', 'sigma', 'n'])


def inv_softplus(x):
    return np.log(1 - np.exp(-x)) + x


def inv_sigmoid(x):
    return np.log(x) - np.log(1. - x)


class GaussianRecurrentLayer(object):
    def __init__(self, ndim, name,
                 corr_init=0.1,
                 learn_variance=True, learn_covariance=True):
        self.seed_rng = np.random.RandomState(42)
        self.name = name
        self.ndim = ndim

        with tf1.variable_scope(name):
            self.mu = tf.zeros((1, ndim), name='prior_mean')

            self.var_vbl = tf1.get_variable("prior_var", (1, ndim), tf.float32,
                                            tf.constant_initializer(inv_softplus(np.sqrt(1.))),
                                            trainable=learn_variance)
            self.var = tf.square(tf.nn.softplus(self.var_vbl))

            self.corr_vbl = tf1.get_variable("prior_corr", (1, ndim), tf.float32,
                                             tf.constant_initializer(inv_sigmoid(corr_init)),
                                             trainable=learn_covariance)
            self.corr = tf.sigmoid(self.corr_vbl)
            self.cov = tf.sigmoid(self.corr_vbl) * self.var

            self.n = tf.zeros([], name='n')
            self.prior = State(self.mu, self.var, self.n)
            self.current_state = self.prior

    @property
    def variables(self):
        return self.mu, self.var_vbl, self.corr_vbl

    def reset(self):
        self.current_state = self.prior
        return self.current_state

    def replicate_state(self, batch_size):
        self.current_state = State(tf.tile(self.mu, [batch_size, 1]), self.var, self.n)
        return self.current_state

    def set_state(self, state):
        self.current_state = state

    def update_distribution(self, observation):
        mu, sigma, n = self.current_state
        n += 1
        dd = self.cov / (self.var + self.cov * (n - 1.))
        mu_out = (1. - dd) * mu + observation * dd
        var_out = (1. - dd) * sigma + (self.var - self.cov) * dd

        self.current_state = State(mu_out, var_out, n)

    def bulk_update_distribution(self, observations):
        mu, sigma, n = self.current_state
        n += tf.cast(tf.shape(observations)[1], dtype=tf.float32)
        mu_out = self.cov / (self.var + self.cov * (n - 1)) * tf.reduce_sum(observations - self.mu, axis=1) + self.mu
        var_out = self.var - n * tf.square(self.cov) / (self.var + self.cov * (n - 1))
        self.current_state = State(mu_out, var_out, n)

    def get_updated_state(self, state, observation):
        mu, sigma, n = state
        n += 1
        dd = self.cov / (self.var + self.cov * (n - 1.))
        mu_out = (1. - dd) * mu + observation * dd
        var_out = (1. - dd) * sigma + (self.var - self.cov) * dd
        return State(mu_out, var_out, n)

    def get_posterior_params(self):
        mu, sigma, _ = self.current_state
        return mu, tf.tile(sigma - (self.var - self.cov), [tf.shape(mu)[0], 1])

    def get_posterior_params_given_state(self, state):
        mu, sigma, _ = state
        return mu, sigma - (self.var - self.cov)

    def get_log_likelihood(self, observation):
        mu, var, _ = self.current_state
        log_pdf = -0.5 * tf.log(2. * np.pi * var) - tf.square(observation - mu) / (2. * var)
        return tf.reduce_sum(log_pdf, -1)

    def get_sequence_log_likelihood(self, observation):
        mu, var, _ = self.current_state
        log_pdf = -0.5 * tf.log(2. * np.pi * var) - tf.square(observation - mu[:, None, :]) / (2. * var)
        return tf.reduce_sum(log_pdf, -1)

    def get_log_likelihood_given_state(self, observation, state):
        mu, var, _ = state
        log_pdf = -0.5 * tf.log(2. * np.pi * var) - tf.square(observation - mu) / (2. * var)
        return tf.reduce_sum(log_pdf, -1)

    def get_factorized_log_likelihood(self, observation, x1_ndim):
        mu, var, _ = self.current_state
        log_pdf = -0.5 * tf.log(2. * np.pi * var) - tf.square(observation - mu) / (2. * var)
        return tf.reduce_sum(log_pdf[:, :x1_ndim], axis=-1), tf.reduce_sum(log_pdf[:, x1_ndim:], axis=-1)

    def get_factorized_log_likelihood_given_state(self, observation, x1_ndim, state):
        mu, var, _ = state
        log_pdf = -0.5 * tf.log(2. * np.pi * var) - tf.square(observation - mu) / (2. * var)
        return tf.reduce_sum(log_pdf[:, :x1_ndim], axis=-1), tf.reduce_sum(log_pdf[:, x1_ndim:], axis=-1)

    def get_log_likelihood_under_prior(self, observation):
        mu, var, _ = self.prior
        log_pdf = -0.5 * tf.log(2. * np.pi * var) - tf.square(observation - mu) / (2. * var)
        return tf.reduce_sum(log_pdf, -1)

    def sample(self, nr_samples=1):
        mu, var, _ = self.current_state
        if nr_samples == 1:
            return mu + tf.sqrt(var) * tf.random_normal(shape=tf.shape(mu), seed=self.seed_rng.randint(317070),
                                                        name="Normal_sampler")
        else:
            return mu[None, :, :] + tf.sqrt(var[None, :, :]) * tf.random_normal(
                shape=(nr_samples, tf.shape(mu)[0], self.ndim),
                seed=self.seed_rng.randint(317070),
                name="Normal_sampler")

    # def sample_given_state(self, state, n_samples=1):
    #     mu, var, _ = state
    #     if n_samples == 1:
    #         return mu + tf.sqrt(var) * tf.random_normal(shape=tf.shape(mu), seed=self.seed_rng.randint(317070),
    #                                                     name="Normal_sampler")
    #     else:
    #         return mu[:, :, None] + tf.sqrt(var[:, :, None]) * tf.random_normal(
    #             shape=(tf.shape(mu)[0], self.ndim, n_samples),
    #             seed=self.seed_rng.randint(317070),
    #             name="Normal_sampler")
