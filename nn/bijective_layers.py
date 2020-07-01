import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1

MAX_LOGSCALE = 2.5


class Orthogonal(object):
    """
    Lasagne orthogonal init from OpenAI
    """

    def __init__(self, scale=1.):
        self.scale = scale

    def __call__(self, shape, dtype=None, partition_info=None):
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (self.scale * q[:shape[0], :shape[1]]).astype(np.float32)

    def get_config(self):
        return {
            'scale': self.scale
        }


def int_shape(x):
    return list(x.get_shape())


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
                                            reverse_inputs_order=False if i % 2 == 0 else True,
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

            if condition is not None:
                condition = dense(condition, num_units=self.n_units, name='condition_layer',
                                  nonlinearity=self.nonlinearity)

            y = masked_dense(input, num_units=self.n_units, num_blocks=self.input_dim,
                             activation=self.nonlinearity,
                             kernel_initializer=Orthogonal(),
                             bias_initializer=tf.constant_initializer(0.01),
                             exclusive_mask=True,
                             condition=condition,
                             name='d1')

            # y = masked_dense(y, num_units=self.n_units, num_blocks=self.input_dim,
            #                  activation=self.nonlinearity,
            #                  exclusive_mask=False,
            #                  kernel_initializer=Orthogonal(),
            #                  bias_initializer=tf.constant_initializer(0.01),
            #                  condition=condition,
            #                  name='d2', )

            l_scale = masked_dense(y, num_units=self.input_dim, num_blocks=self.input_dim,
                                   activation=lambda x: MAX_LOGSCALE * tf.tanh(x),
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
        if condition is not None:
            condition = dense_wn(condition, num_units=self.n_units, name='condition_layer',
                                 activation=self.nonlinearity)

        y = masked_dense_wn(input, num_units=self.n_units, num_blocks=self.input_dim,
                            activation=self.nonlinearity,
                            exclusive_mask=True,
                            condition=condition,
                            name='d1')

        # y = masked_dense_wn(y, num_units=self.n_units, num_blocks=self.input_dim,
        #                     activation=self.nonlinearity,
        #                     exclusive_mask=False,
        #                     condition=condition,
        #                     name='d2')

        l_scale = masked_dense(y, num_units=self.input_dim, num_blocks=self.input_dim,
                               activation=lambda x: MAX_LOGSCALE * tf.tanh(x),
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
        with tf1.variable_scope(self.name, reuse=tf1.AUTO_REUSE):
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
            g = tf1.get_variable(name='g', shape=[num_units], dtype=tf.float32,
                                 initializer=tf.constant_initializer(np.sqrt(2. * fan_in / num_units)),
                                 trainable=True)
            b = tf1.get_variable(name='b', shape=[num_units], dtype=tf.float32,
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
            V = tf1.get_variable(name='V', shape=[fan_in, num_units], dtype=tf.float32,
                                 initializer=Orthogonal(), trainable=True)
            g = tf1.get_variable(name='g', shape=[num_units], dtype=tf.float32,
                                 initializer=tf.constant_initializer(np.sqrt(2. * fan_in / num_units)),
                                 trainable=True)
            b = tf1.get_variable(name='b', shape=[num_units], dtype=tf.float32,
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


class NVPLayer:
    def __init__(self, mask_type, input_size, name='NVPLayer', nonlinearity=tf.nn.relu, n_units=1024,
                 weight_norm=True, noise_dim=0):
        self.mask_type = mask_type
        self.name = name
        self.nonlinearity = nonlinearity
        self.n_units = n_units
        self.weight_norm = weight_norm
        self.input_size = input_size
        self.noise_dim = noise_dim

    def get_mask(self, xs, mask_type):

        assert self.mask_type in ['even', 'odd', 'noise_dec', 'noise_enc']

        ndim = self.input_size

        b = tf.range(ndim)
        if 'even' in mask_type:
            b = tf.cast(tf.mod(b, 2), tf.float32)
        elif 'odd' in mask_type:
            b = 1. - tf.cast(tf.mod(b, 2), tf.float32)
        elif 'noise_enc' in mask_type:
            b = tf.concat([tf.ones([ndim - self.noise_dim]), tf.zeros([self.noise_dim])], axis=-1)
        elif 'noise_dec' in mask_type:
            b = tf.concat([tf.zeros([ndim - self.noise_dim]), tf.ones([self.noise_dim])], axis=-1)

        b_mask = tf.ones((xs[0], ndim))
        b_mask = b_mask * b

        b_mask = tf.reshape(b_mask, xs)

        return b_mask

    def function_s_t(self, x, mask, condition, name='function_s_t_dense'):
        if self.weight_norm:
            return self.function_s_t_wn(x, mask, condition, name + '_wn')
        else:
            with tf.variable_scope(name):
                xs = tf.shape(x)
                y = tf.reshape(x, (xs[0], -1))
                ndim = tf.shape(y)[-1]

                y = dense(y, num_units=self.n_units, nonlinearity=self.nonlinearity,
                          kernel_initializer=Orthogonal(),
                          bias_initializer=tf.constant_initializer(0.01), name='d1', condition=condition)
                y = dense(y, num_units=self.n_units, nonlinearity=self.nonlinearity,
                          kernel_initializer=Orthogonal(),
                          bias_initializer=tf.constant_initializer(0.01), name='d2', condition=condition)

                l_scale = dense(y, num_units=ndim,
                                nonlinearity=lambda x: MAX_LOGSCALE * tf.tanh(x),
                                kernel_initializer=tf.constant_initializer(0.),
                                bias_initializer=tf.constant_initializer(0.), name='d_scale', condition=condition)
                l_scale = tf.reshape(l_scale, shape=xs)
                l_scale *= 1 - mask

                m_translation = dense(y, num_units=ndim, nonlinearity=None,
                                      kernel_initializer=tf.constant_initializer(0.),
                                      bias_initializer=tf.constant_initializer(0.), name='d_translate',
                                      condition=condition)
                m_translation = tf.reshape(m_translation, shape=xs)
                m_translation *= 1 - mask

                return l_scale, m_translation

    def function_s_t_wn(self, x, mask, condition, name):
        with tf.variable_scope(name):
            y = dense_wn(x, num_units=self.n_units, name='d1', activation=self.nonlinearity, condition=condition)
            y = dense_wn(y, num_units=self.n_units, name='d2', activation=self.nonlinearity, condition=condition)

            l_scale = dense(y, num_units=self.input_size,
                            nonlinearity=lambda x: MAX_LOGSCALE * tf.tanh(x),
                            kernel_initializer=tf.constant_initializer(0.),
                            bias_initializer=tf.constant_initializer(0.), name='d_scale', condition=condition)
            l_scale *= 1 - mask

            m_translation = dense(y, num_units=self.input_size, nonlinearity=None,
                                  kernel_initializer=tf.constant_initializer(0.),
                                  bias_initializer=tf.constant_initializer(0.), name='d_translate',
                                  condition=condition)
            m_translation *= 1 - mask

            return l_scale, m_translation

    def forward_and_jacobian(self, x, sum_log_det_jacobians, condition=None):
        with tf1.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            xs = tf.shape(x)
            b = self.get_mask(xs, self.mask_type)

            # masked half of x
            x1 = x * b
            l, m = self.function_s_t(x1, b, condition)
            y = x1 + tf.multiply(1. - b, x * tf.exp(l) + m)
            log_det_jacobian = tf.reduce_sum(l, 1)
            sum_log_det_jacobians += log_det_jacobian

            return y, sum_log_det_jacobians

    def backward(self, y, sum_log_det_jacobians=None, condition=None):
        with tf1.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            ys = tf.shape(y)
            b = self.get_mask(ys, self.mask_type)

            y1 = y * b
            l, m = self.function_s_t(y1, b, condition)
            x = y1 + tf.multiply(y * (1. - b) - m, tf.exp(-l))

            if sum_log_det_jacobians is not None:
                log_det_jacobian = -1. * tf.reduce_sum(l, 1)
                sum_log_det_jacobians += log_det_jacobian
                return x, sum_log_det_jacobians
            else:
                return x


class NVP:
    def __init__(self, input_size, n_maf_layers, name='NVP', nonlinearity=tf.nn.relu, n_units=32,
                 weight_norm=True, debug_mode=None, noise_dim=None, tie_weights=False):

        self.input_dim = input_size
        self.n_nvp_layers = n_maf_layers

        self.nvp_layers = []

        for i in range(n_maf_layers):
            if noise_dim is None:
                mask_type = 'even' if i % 2 == 0 else 'odd'
            else:
                mask_type = 'noise_enc' if i % 2 == 0 else 'noise_dec'

            if not tie_weights:
                layer_name = '%s_%s' % (name, i)
            else:
                layer_name = '%s_%s' % (name, mask_type)

            self.nvp_layers.append(NVPLayer(input_size=input_size, name=layer_name,
                                            nonlinearity=nonlinearity, n_units=n_units,
                                            mask_type=mask_type,
                                            weight_norm=weight_norm, noise_dim=noise_dim))

    def forward_and_jacobian(self, x, sum_log_det_jacobians, condition=None):
        for i in range(self.n_nvp_layers):
            x, sum_log_det_jacobians = self.nvp_layers[i].forward_and_jacobian(x, sum_log_det_jacobians, condition)
        return x, sum_log_det_jacobians

    def backward(self, x, sum_log_det_jacobians=None, condition=None):
        if sum_log_det_jacobians is None:
            for i in reversed(range(self.n_nvp_layers)):
                x = self.nvp_layers[i].backward(x, sum_log_det_jacobians, condition)
            return x
        else:
            for i in reversed(range(self.n_nvp_layers)):
                x, sum_log_det_jacobians = self.nvp_layers[i].backward(x, sum_log_det_jacobians, condition)
            return x, sum_log_det_jacobians
