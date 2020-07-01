import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_probability as tfp

from nn.bijective_layers import MAF, NVP
from nn.gp_layer import GaussianRecurrentLayer
from nn.tp_layer import StudentRecurrentLayer

tfd = tfp.distributions


class BrunoNet(object):
    def __init__(self, x_dim=1, y_dim=1, name='bruno',
                 maf_num_hidden=32, n_maf_layers=2,
                 weight_norm=True, debug_mode=False, extra_dims=0, corr_init=0.1,
                 learn_covariance=True, learn_variance=True,
                 maf_nonlinearity=tf.nn.relu,
                 noise_distribution='uniform',
                 bijection='maf',
                 process='gp'):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.name = name
        self.extra_dims = extra_dims
        self.noise = noise_distribution
        self.max_context_len = 30

        if noise_distribution == 'gaussian':
            self.noise_entropy = 0.5 * extra_dims * np.log(2 * np.pi * np.e)
        elif noise_distribution == 'uniform':
            self.noise_entropy = extra_dims * np.log(1.)
        else:
            ValueError('wrong noise distribution')
        print('noise entropy', self.noise_entropy)

        self.latent_ndim = y_dim + extra_dims

        self.standard_mvn = tfd.MultivariateNormalDiag(loc=[0] * extra_dims,
                                                       scale_diag=[1] * extra_dims)

        with tf1.variable_scope(name):
            if bijection == 'maf':
                self.maf = MAF(input_size=self.latent_ndim, name=name + '/model/maf',
                               n_maf_layers=n_maf_layers,
                               n_units=maf_num_hidden, weight_norm=weight_norm, debug_mode=debug_mode,
                               nonlinearity=maf_nonlinearity)
            elif bijection == 'nvp':
                self.maf = NVP(input_size=self.latent_ndim, name=name + '/model/nvp',
                               n_maf_layers=n_maf_layers,
                               n_units=maf_num_hidden, weight_norm=weight_norm, debug_mode=debug_mode,
                               nonlinearity=maf_nonlinearity,
                               noise_dim=extra_dims)
            if process == 'gp':
                self.gp_layer = GaussianRecurrentLayer(ndim=self.latent_ndim, corr_init=corr_init,
                                                       learn_covariance=learn_covariance,
                                                       learn_variance=learn_variance, name='model/gp')
            elif process == 'tp':
                self.gp_layer = StudentRecurrentLayer(ndim=self.latent_ndim, corr_init=corr_init,
                                                      name='model/gp')

            self.prior = self.gp_layer.prior

    def reset(self):
        return self.gp_layer.reset()

    def loss(self, query, target_y):

        (_, _), target_x = query

        target_z, target_jacob = self.encode(target_x, target_y)

        context_len = tf.random.uniform(shape=[], minval=3, maxval=self.max_context_len, dtype=tf.int32)
        self.gp_layer.reset()
        self.gp_layer.bulk_update_distribution(target_z[:, :context_len, :])
        log_probs = self.gp_layer.get_sequence_log_likelihood(target_z) + target_jacob
        loss_target = -tf.reduce_mean(log_probs)
        return loss_target

    def test_median(self, query, n_context):

        (context_x, context_y), target_x = query

        context_z, _ = self.encode(context_x, context_y)

        medians = []
        for nc in n_context:
            self.gp_layer.reset()
            self.gp_layer.bulk_update_distribution(context_z[:, :nc])

            mu, var, _ = self.gp_layer.current_state
            mu = tf.tile(mu[:, None, :], [1, tf.shape(target_x)[1], 1])
            median, _ = self.decode(target_x, mu)
            medians.append(median)

        medians = tf.concat(medians, axis=0)

        return medians, medians * 0.

    def test_sample_mean(self, query, n_context, n_curves=100):

        (context_x, context_y), target_x = query

        context_z, _ = self.encode(context_x, context_y)

        means, stds = [], []
        for nc in n_context:
            self.gp_layer.reset()
            self.gp_layer.bulk_update_distribution(context_z[:, :nc])

            z_samples = self.gp_layer.sample(nr_samples=400 * n_curves)
            z_samples = tf.reshape(z_samples, (n_curves, 400, self.latent_ndim))

            target_x_tiled = tf.tile(target_x, [n_curves, 1, 1])

            y_samples, _ = self.decode(target_x_tiled, z_samples)

            y_mean = tf.reduce_mean(y_samples, axis=0, keepdims=True)
            y_std = tf.math.reduce_std(y_samples, axis=0, keepdims=True)

            means.append(y_mean)
            stds.append(y_std)

        means = tf.concat(means, axis=0)
        stds = tf.concat(stds, axis=0)

        return means, stds

    def get_gp_state(self, query):

        (context_x, context_y), target_x = query
        context_z, _ = self.encode(context_x, context_y)

        self.gp_layer.reset()
        self.gp_layer.bulk_update_distribution(context_z)

        return self.gp_layer.current_state

    def encode(self, x, y):

        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        seq_len = input_shape[1] if input_shape.shape == 3 else 1

        x = tf.reshape(x, (batch_size * seq_len, self.x_dim))
        y = tf.reshape(y, (batch_size * seq_len, self.y_dim))

        condition = x
        input = y

        if self.extra_dims > 0:
            if self.noise == 'gaussian':
                noise = tf.random_normal(shape=(batch_size * seq_len, self.extra_dims))
            elif self.noise == 'uniform':
                noise = tf.random_uniform(shape=(batch_size * seq_len, self.extra_dims))
            else:
                raise ValueError('wrong noise')
            input = tf.concat([input, noise], axis=-1)

        jacob = tf.zeros(batch_size * seq_len)

        z, jacob = self.maf.forward_and_jacobian(input, jacob, condition=condition)

        # reshape sequences into their original shape (batch_size, seq_len, input_dim)
        if input_shape.shape == 3:
            z = tf.reshape(z, shape=(batch_size, seq_len, self.latent_ndim))
            jacob = tf.reshape(jacob, (batch_size, seq_len))

        return z, jacob

    def decode(self, x, z):

        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        seq_len = input_shape[1] if input_shape.shape == 3 else 1

        x = tf.reshape(x, (batch_size * seq_len, self.x_dim))
        z = tf.reshape(z, (batch_size * seq_len, self.latent_ndim))

        condition = x
        bwd_jacob = tf.zeros(batch_size * seq_len)

        output, neq_jacob = self.maf.backward(z, bwd_jacob, condition=condition)

        if self.extra_dims > 0:
            output = output[:, :-self.extra_dims]

        # reshape sequences into their original shape (batch_size, seq_len, input_dim)
        if input_shape.shape == 3:
            output = tf.reshape(output, shape=(batch_size, seq_len, self.y_dim))
            bwd_jacob = tf.reshape(bwd_jacob, (batch_size, seq_len))

        return output, bwd_jacob
