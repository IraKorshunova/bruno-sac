import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1

from utils.nn_utils import Orthogonal, int_shape


class MAF:
    def __init__(self, input_size, n_maf_layers, name='MAF', nonlinearity=tf.nn.relu, n_units=32, weight_norm=True,
                 debug_mode=False):

        assert input_size < n_units

        self.input_dim = input_size
        self.n_maf_layers = n_maf_layers

        self.maf_layers = []

        for i in range(n_maf_layers):
            self.maf_layers.append(MAFLayer(input_size=input_size, name='%s_%s' % (name, i),
                                            nonlinearity=nonlinearity, n_units=n_units,
                                            reverse_inputs_order=True if i % 2 == 0 else False,
                                            weight_norm=weight_norm, debug_mode=debug_mode))

    def forward_and_jacobian(self, x, sum_log_det_jacobians, condition=None):
        for i in range(self.n_maf_layers):
            x, sum_log_det_jacobians = self.maf_layers[i].forward_and_jacobian(x, sum_log_det_jacobians, condition)
        return x, sum_log_det_jacobians

    def backward(self, x, sum_log_det_jacobians=None, condition=None):
        if sum_log_det_jacobians is None:
            for i in reversed(range(self.n_maf_layers)):
                x = self.maf_layers[i].backward(x, sum_log_det_jacobians, condition)
            return x
        else:
            for i in reversed(range(self.n_maf_layers)):
                x, sum_log_det_jacobians = self.maf_layers[i].backward(x, sum_log_det_jacobians, condition)
            return x, sum_log_det_jacobians


class MAFLayer:
    def __init__(self, input_size, name='MAFLayer', nonlinearity=tf.nn.relu, n_units=32,
                 reverse_inputs_order=False, weight_norm=True, debug_mode=False):
        self.input_dim = input_size
        self.name = name
        self.nonlinearity = nonlinearity
        self.n_units = n_units
        self.weight_norm = weight_norm
        self.reverse_inputs_order = reverse_inputs_order
        self.debug_mode = debug_mode

        assert self.input_dim < self.n_units

    def function_s_t(self, input, condition):
        if self.weight_norm:
            return self.function_s_t_wn(input, condition)
        else:

            condition = dense(condition, num_units=self.n_units, name='condition_layer',
                              nonlinearity=self.nonlinearity)
            y = masked_dense(input, num_units=self.n_units, num_blocks=self.input_dim,
                             activation=self.nonlinearity,
                             kernel_initializer=Orthogonal(),
                             bias_initializer=tf.constant_initializer(0.01),
                             exclusive_mask=True,
                             condition=condition,
                             name='d1')

            l_scale = masked_dense(y, num_units=self.input_dim, num_blocks=self.input_dim,
                                   activation=tf.tanh,
                                   exclusive_mask=False,
                                   kernel_initializer=tf.constant_initializer(
                                       0.) if not self.debug_mode else Orthogonal(),
                                   bias_initializer=tf.constant_initializer(0.),
                                   condition=condition,
                                   name='d_scale')

            m_translation = masked_dense(y, num_units=self.input_dim, num_blocks=self.input_dim,
                                         activation=None,
                                         exclusive_mask=False,
                                         kernel_initializer=tf.constant_initializer(0.),
                                         bias_initializer=tf.constant_initializer(0.),
                                         condition=condition,
                                         name='d_translate')

            return l_scale, m_translation

    def function_s_t_wn(self, input, condition):

        condition = dense_wn(condition, num_units=self.n_units, name='condition_layer',
                             activation=self.nonlinearity)

        y = masked_dense_wn(input, num_units=self.n_units, num_blocks=self.input_dim,
                            activation=self.nonlinearity,
                            exclusive_mask=True,
                            condition=condition,
                            name='d1')

        l_scale = masked_dense(y, num_units=self.input_dim, num_blocks=self.input_dim,
                               activation=tf.tanh,
                               exclusive_mask=False,
                               kernel_initializer=tf.constant_initializer(0.) if not self.debug_mode else Orthogonal(),
                               bias_initializer=tf.constant_initializer(0.), condition=condition,
                               name='d_scale')

        m_translation = masked_dense(y, num_units=self.input_dim, num_blocks=self.input_dim,
                                     activation=None,
                                     exclusive_mask=False,
                                     kernel_initializer=tf.constant_initializer(0.),
                                     bias_initializer=tf.constant_initializer(0.),
                                     condition=condition,
                                     name='d_translate')

        return l_scale, m_translation

    def forward_and_jacobian(self, x, sum_log_det_jacobians, condition=None):
        with tf1.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            if self.reverse_inputs_order:
                x = tf.reverse(x, axis=[-1])

            log_scale, translation = self.function_s_t(input=x, condition=condition)
            sum_log_det_jacobians -= tf.reduce_sum(log_scale, 1)
            y = (x - translation) / tf.exp(log_scale)

            return y, sum_log_det_jacobians

    def backward(self, x, sum_log_det_jacobians=None, condition=None):
        with tf1.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            y = tf.zeros_like(x, name=self.name + 'y0')
            for i in range(self.input_dim):
                log_scale, translation = self.function_s_t(input=y, condition=condition)
                y = x * tf.exp(log_scale) + translation
                if sum_log_det_jacobians is not None:
                    sum_log_det_jacobians += log_scale[:, i]

            if self.reverse_inputs_order:
                y = tf.reverse(y, axis=[-1])
            if sum_log_det_jacobians is not None:
                return y, sum_log_det_jacobians
            else:
                return y


def gen_slices(num_blocks, n_in, n_out, exclusive_mask):
    slices = []
    col = 0
    d_in = n_in // num_blocks
    d_out = n_out // num_blocks
    row = d_out if exclusive_mask else 0
    for _ in range(num_blocks):
        row_slice = slice(row, None)
        col_slice = slice(col, col + d_in)
        slices.append([row_slice, col_slice])
        col += d_in
        row += d_out
    return slices


def generate_mask(num_blocks, n_in, n_out, exclusive_mask, dtype=tf.float32):
    mask = np.zeros([n_out, n_in], dtype=dtype.as_numpy_dtype())
    slices = gen_slices(num_blocks, n_in, n_out, exclusive_mask=exclusive_mask)
    for [row_slice, col_slice] in slices:
        mask[row_slice, col_slice] = 1
    return mask


def masked_dense(x, num_units, num_blocks, exclusive_mask, name, activation=None,
                 kernel_initializer=Orthogonal(),
                 bias_initializer=tf.constant_initializer(0.), condition=None):
    with tf1.variable_scope(name):

        input_dim = int_shape(x)[-1]
        mask = generate_mask(num_blocks, input_dim, num_units, exclusive_mask).T

        def masked_initializer(shape, dtype=None, partition_info=None):
            return mask * kernel_initializer(shape, dtype, partition_info)

        if condition is None:
            output = tf1.layers.dense(x, units=num_units, activation=activation,
                                      kernel_initializer=masked_initializer,
                                      kernel_constraint=lambda x: mask * x,
                                      bias_initializer=bias_initializer, name='masked_dense')
            return output
        else:
            ndim_condition = int_shape(condition)[-1]
            o1 = tf.concat([condition, x], axis=-1)
            mask = np.concatenate([np.ones((ndim_condition, num_units)), mask], axis=0)
            output = tf1.layers.dense(o1, units=num_units, activation=None,
                                      bias_initializer=bias_initializer,
                                      kernel_initializer=masked_initializer,
                                      kernel_constraint=lambda x: mask * x,
                                      name='masked_dense')

            if activation is not None:
                output = activation(output)
            return output


def masked_dense_wn(x, num_units, num_blocks, exclusive_mask, name, activation=None, use_bias=True,
                    condition=None, mask=None,
                    kernel_initializer=Orthogonal(),
                    eps=1e-12):
    """
    Weight norm with initialization from Arpit et al., 2019
    """
    with tf1.variable_scope(name):

        input_dim = int_shape(x)[-1]
        if mask is None:
            mask = generate_mask(num_blocks, input_dim, num_units, exclusive_mask).T

        def masked_initializer(shape, dtype=None, partition_info=None):
            return mask * kernel_initializer(shape, dtype, partition_info)

        if condition is None:
            fan_in = int(x.get_shape()[1])
            V = mask * tf.get_variable(name='V', shape=[input_dim, num_units], dtype=tf.float32,
                                       initializer=masked_initializer, trainable=True)
            g = tf.get_variable(name='g', shape=[num_units], dtype=tf.float32,
                                initializer=tf.constant_initializer(np.sqrt(2. * fan_in / num_units)),
                                trainable=True)
            b = tf.get_variable(name='b', shape=[num_units], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.), trainable=use_bias)

            x = tf.matmul(x, V)
            scaler = g / tf.norm(V + eps, axis=0)
            x = tf.reshape(scaler, [1, num_units]) * x + tf.reshape(b, [1, num_units])

            if activation is not None:
                x = activation(x)
            return x
        else:
            ndim_condition = int_shape(condition)[-1]
            o1 = tf.concat([condition, x], axis=-1)
            mask = np.concatenate([np.ones((ndim_condition, num_units)), mask], axis=0)
            output = masked_dense_wn(o1, num_units=num_units, num_blocks=num_blocks,
                                     exclusive_mask=exclusive_mask, activation=None, use_bias=True,
                                     name='masked_dense_wn',
                                     mask=mask)

            if activation is not None:
                output = activation(output)
            return output


def dense(x, num_units, name, nonlinearity=None, kernel_initializer=Orthogonal(),
          bias_initializer=tf.constant_initializer(0.), condition=None):
    with tf.variable_scope(name):
        if condition is None:
            return tf.layers.dense(x, units=num_units, activation=nonlinearity,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer, name=name)
        else:
            ndim = int_shape(x)[-1]
            h = tf.layers.dense(condition, units=ndim, activation=tf.nn.leaky_relu,
                                bias_initializer=bias_initializer,
                                kernel_initializer=Orthogonal(), name='label')
            o1 = tf.concat([h, x], axis=-1)
            output = tf.layers.dense(o1, units=num_units, activation=None,
                                     bias_initializer=bias_initializer,
                                     kernel_initializer=kernel_initializer, name='dense')

            if nonlinearity is not None:
                output = nonlinearity(output)
            return output


def dense_wn(x, num_units, name, activation=None, use_bias=True, condition=None, eps=1e-12):
    """
    Weight norm with initialization from Arpit et al., 2019
    """
    with tf1.variable_scope(name):
        if condition is None:
            fan_in = int(x.get_shape()[1])
            num_units = num_units if isinstance(num_units, int) else num_units.value
            V = tf.get_variable(name='V', shape=[fan_in, num_units], dtype=tf.float32,
                                initializer=Orthogonal(), trainable=True)
            g = tf.get_variable(name='g', shape=[num_units], dtype=tf.float32,
                                initializer=tf.constant_initializer(np.sqrt(2. * fan_in / num_units)),
                                trainable=True)
            b = tf.get_variable(name='b', shape=[num_units], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.), trainable=use_bias)

            x = tf.matmul(x, V)
            scaler = g / tf.norm(V + eps, axis=0)
            x = tf.reshape(scaler, [1, num_units]) * x + tf.reshape(b, [1, num_units])

            if activation is not None:
                x = activation(x)
            return x
        else:
            ndim = int_shape(x)[-1]
            h = dense_wn(condition, num_units=ndim, activation=tf.nn.relu, use_bias=True, name='label')

            o1 = tf.concat([h, x], axis=-1)
            output = dense_wn(o1, num_units=num_units, activation=None, use_bias=True, name='dense')

            if activation is not None:
                output = activation(output)
            return output
