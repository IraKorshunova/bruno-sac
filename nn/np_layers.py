import tensorflow as tf


# utility methods
def batch_mlp(input, output_sizes, variable_scope):
    """Apply MLP to the final axis of a 3D tensor (reusing already defined MLPs).

    Args:
      input: input tensor of shape [B,n,d_in].
      output_sizes: An iterable containing the output sizes of the MLP as defined
          in `basic.Linear`.
      variable_scope: String giving the name of the variable scope. If this is set
          to be the same as a previously defined MLP, then the weights are reused.

    Returns:
      tensor of shape [B,n,d_out] where d_out=output_sizes[-1]
    """
    # Get the shapes of the input and reshape to parallelise across observations
    batch_size, _, filter_size = input.shape.as_list()
    output = tf.reshape(input, (-1, filter_size))
    output.set_shape((None, filter_size))

    # Pass through MLP
    with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
        for i, size in enumerate(output_sizes[:-1]):
            output = tf.nn.relu(
                tf.layers.dense(output, size, name="layer_{}".format(i)))

        # Last layer without a ReLu
        output = tf.layers.dense(
            output, output_sizes[-1], name="layer_{}".format(i + 1))

    # Bring back into original shape
    output = tf.reshape(output, (batch_size, -1, output_sizes[-1]))
    return output


class DeterministicEncoder(object):
    """The Deterministic Encoder."""

    def __init__(self, output_sizes, attention):
        """(A)NP deterministic encoder.

        Args:
          output_sizes: An iterable containing the output sizes of the encoding MLP.
          attention: The attention module.
        """
        self._output_sizes = output_sizes
        self._attention = attention

    def __call__(self, context_x, context_y, target_x):
        """Encodes the inputs into one representation.

        Args:
          context_x: Tensor of shape [B,observations,d_x]. For this 1D regression
              task this corresponds to the x-values.
          context_y: Tensor of shape [B,observations,d_y]. For this 1D regression
              task this corresponds to the y-values.
          target_x: Tensor of shape [B,target_observations,d_x].
              For this 1D regression task this corresponds to the x-values.

        Returns:
          The encoded representation. Tensor of shape [B,target_observations,d]
        """

        # Concatenate x and y along the filter axes
        encoder_input = tf.concat([context_x, context_y], axis=-1)

        # Pass final axis through MLP
        hidden = batch_mlp(encoder_input, self._output_sizes,
                           "deterministic_encoder")

        # Apply attention
        with tf.variable_scope("deterministic_encoder", reuse=tf.AUTO_REUSE):
            hidden = self._attention(context_x, target_x, hidden)

        return hidden


class LatentEncoder(object):
    """The Latent Encoder."""

    def __init__(self, output_sizes, num_latents):
        """(A)NP latent encoder.

        Args:
          output_sizes: An iterable containing the output sizes of the encoding MLP.
          num_latents: The latent dimensionality.
        """
        self._output_sizes = output_sizes
        self._num_latents = num_latents

    def __call__(self, x, y):
        """Encodes the inputs into one representation.

        Args:
          x: Tensor of shape [B,observations,d_x]. For this 1D regression
              task this corresponds to the x-values.
          y: Tensor of shape [B,observations,d_y]. For this 1D regression
              task this corresponds to the y-values.

        Returns:
          A normal distribution over tensors of shape [B, num_latents]
        """

        # Concatenate x and y along the filter axes
        encoder_input = tf.concat([x, y], axis=-1)

        # Pass final axis through MLP
        hidden = batch_mlp(encoder_input, self._output_sizes, "latent_encoder")

        # Aggregator: take the mean over all points
        hidden = tf.reduce_mean(hidden, axis=1)

        # Have further MLP layers that map to the parameters of the Gaussian latent
        with tf.variable_scope("latent_encoder", reuse=tf.AUTO_REUSE):
            # First apply intermediate relu layer
            hidden = tf.nn.relu(
                tf.layers.dense(hidden,
                                (self._output_sizes[-1] + self._num_latents) / 2,
                                name="penultimate_layer"))
            # Then apply further linear layers to output latent mu and log sigma
            mu = tf.layers.dense(hidden, self._num_latents, name="mean_layer")
            log_sigma = tf.layers.dense(hidden, self._num_latents, name="std_layer")

        # Compute sigma
        sigma = 0.1 + 0.9 * tf.sigmoid(log_sigma)

        return tf.contrib.distributions.Normal(loc=mu, scale=sigma)


class Decoder(object):
    """The Decoder."""

    def __init__(self, output_sizes):
        """(A)NP decoder.

        Args:
          output_sizes: An iterable containing the output sizes of the decoder MLP
              as defined in `basic.Linear`.
        """
        self._output_sizes = output_sizes

    def __call__(self, representation, target_x):
        """Decodes the individual targets.

        Args:
          representation: The representation of the context for target predictions.
              Tensor of shape [B,target_observations,?].
          target_x: The x locations for the target query.
              Tensor of shape [B,target_observations,d_x].

        Returns:
          dist: A multivariate Gaussian over the target points. A distribution over
              tensors of shape [B,target_observations,d_y].
          mu: The mean of the multivariate Gaussian.
              Tensor of shape [B,target_observations,d_x].
          sigma: The standard deviation of the multivariate Gaussian.
              Tensor of shape [B,target_observations,d_x].
        """
        # concatenate target_x and representation
        hidden = tf.concat([representation, target_x], axis=-1)

        # Pass final axis through MLP
        hidden = batch_mlp(hidden, self._output_sizes, "decoder")

        # Get the mean an the variance
        mu, log_sigma = tf.split(hidden, 2, axis=-1)

        # Bound the variance
        sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)

        # Get the distribution
        dist = tf.contrib.distributions.MultivariateNormalDiag(
            loc=mu, scale_diag=sigma)

        return dist, mu, sigma


class LatentModel(object):
    """The (A)NP model."""

    def __init__(self, latent_encoder_output_sizes, num_latents,
                 decoder_output_sizes, use_deterministic_path=False,
                 deterministic_encoder_output_sizes=None, attention=None,
                 n_test_samples=100):
        """Initialises the model.

        Args:
          latent_encoder_output_sizes: An iterable containing the sizes of hidden
              layers of the latent encoder.
          num_latents: The latent dimensionality.
          decoder_output_sizes: An iterable containing the sizes of hidden layers of
              the decoder. The last element should correspond to d_y * 2
              (it encodes both mean and variance concatenated)
          use_deterministic_path: a boolean that indicates whether the deterministic
              encoder is used or not.
          deterministic_encoder_output_sizes: An iterable containing the sizes of
              hidden layers of the deterministic encoder. The last one is the size
              of the deterministic representation r.
          attention: The attention module used in the deterministic encoder.
              Only relevant when use_deterministic_path=True.
        """
        self._latent_encoder = LatentEncoder(latent_encoder_output_sizes,
                                             num_latents)
        self._decoder = Decoder(decoder_output_sizes)
        self._use_deterministic_path = use_deterministic_path
        if use_deterministic_path:
            self._deterministic_encoder = DeterministicEncoder(
                deterministic_encoder_output_sizes, attention)
        self.n_test_samples = n_test_samples

    def __call__(self, query, num_targets, target_y=None, n_context=None):
        """Returns the predicted mean and variance at the target points.

        Args:
          query: Array containing ((context_x, context_y), target_x) where:
              context_x: Tensor of shape [B,num_contexts,d_x].
                  Contains the x values of the context points.
              context_y: Tensor of shape [B,num_contexts,d_y].
                  Contains the y values of the context points.
              target_x: Tensor of shape [B,num_targets,d_x].
                  Contains the x values of the target points.
          num_targets: Number of target points.
          target_y: The ground truth y values of the target y.
              Tensor of shape [B,num_targets,d_y].

        Returns:
          log_p: The log_probability of the target_y given the predicted
              distribution. Tensor of shape [B,num_targets].
          mu: The mean of the predicted distribution.
              Tensor of shape [B,num_targets,d_y].
          sigma: The variance of the predicted distribution.
              Tensor of shape [B,num_targets,d_y].
        """

        (context_x, context_y), target_x = query

        if n_context is not None:
            context_x = context_x[:, :n_context]
            context_y = context_y[:, :n_context]

        # Pass query through the encoder and the decoder
        prior = self._latent_encoder(context_x, context_y)

        # For training, when target_y is available, use targets for latent encoder.
        # Note that targets contain contexts by design.
        if target_y is None:
            latent_rep = prior.sample(self.n_test_samples)
            latent_rep = tf.tile(latent_rep, [1, num_targets, 1])
            target_x = tf.tile(target_x, [self.n_test_samples, 1, 1])
        # For testing, when target_y unavailable, use contexts for latent encoder.
        else:
            posterior = self._latent_encoder(target_x, target_y)
            latent_rep = posterior.sample()
            latent_rep = tf.tile(tf.expand_dims(latent_rep, axis=1),
                             [1, num_targets, 1])

        dist, mu, sigma = self._decoder(latent_rep, target_x)

        # If we want to calculate the log_prob for training we will make use of the
        # target_y. At test time the target_y is not available so we return None.
        if target_y is not None:
            log_p = dist.log_prob(target_y)
            posterior = self._latent_encoder(target_x, target_y)
            kl = tf.reduce_sum(
                tf.contrib.distributions.kl_divergence(posterior, prior),
                axis=-1, keepdims=True)
            kl = tf.tile(kl, [1, num_targets])
            loss = - tf.reduce_mean(log_p - kl / tf.cast(num_targets, tf.float32))
        else:
            log_p = None
            kl = None
            loss = None

        return mu, sigma, log_p, kl, loss


def uniform_attention(q, v):
    """Uniform attention. Equivalent to np.

    Args:
      q: queries. tensor of shape [B,m,d_k].
      v: values. tensor of shape [B,n,d_v].

    Returns:
      tensor of shape [B,m,d_v].
    """
    total_points = tf.shape(q)[1]
    rep = tf.reduce_mean(v, axis=1, keepdims=True)  # [B,1,d_v]
    rep = tf.tile(rep, [1, total_points, 1])
    return rep


class Attention(object):
    """The Attention module."""

    def __init__(self, rep, output_sizes, att_type, scale=1., normalise=True,
                 num_heads=8):
        """Create attention module.

        Takes in context inputs, target inputs and
        representations of each context input/output pair
        to output an aggregated representation of the context data.
        Args:
          rep: transformation to apply to contexts before computing attention.
              One of: ['identity','mlp'].
          output_sizes: list of number of hidden units per layer of mlp.
              Used only if rep == 'mlp'.
          att_type: type of attention. One of the following:
              ['uniform','laplace','dot_product','multihead']
          scale: scale of attention.
          normalise: Boolean determining whether to:
              1. apply softmax to weights so that they sum to 1 across context pts or
              2. apply custom transformation to have weights in [0,1].
          num_heads: number of heads for multihead.
        """
        self._rep = rep
        self._output_sizes = output_sizes
        self._type = att_type
        self._scale = scale
        self._normalise = normalise
        if self._type == 'multihead':
            self._num_heads = num_heads

    def __call__(self, x1, x2, r):
        """Apply attention to create aggregated representation of r.

        Args:
          x1: tensor of shape [B,n1,d_x].
          x2: tensor of shape [B,n2,d_x].
          r: tensor of shape [B,n1,d].

        Returns:
          tensor of shape [B,n2,d]

        Raises:
          NameError: The argument for rep/type was invalid.
        """
        if self._rep == 'identity':
            k, q = (x1, x2)
        elif self._rep == 'mlp':
            # Pass through MLP
            k = batch_mlp(x1, self._output_sizes, "attention")
            q = batch_mlp(x2, self._output_sizes, "attention")
        else:
            raise NameError("'rep' not among ['identity','mlp']")

        rep = uniform_attention(q, r)

        return rep
