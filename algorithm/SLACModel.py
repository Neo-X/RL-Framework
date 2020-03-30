###
# python3 trainModel.py --config=settings/terrainRLImitate/PPO/SLAC_mini.json -p 4 --bootstrap_samples=1000 --max_epoch_length=16 --rollouts=4 --skip_rollouts=true --train_actor=false --train_critic=false --epochs=32 --fd_updates_per_actor_update=64 --on_policy=fast

import numpy as np
# import lasagne
import sys
from dill.settings import settings
sys.path.append('../')
from model.ModelUtil import *
from model.LearningUtil import loglikelihood, likelihood, likelihoodMEAN, kl_D_keras, entropy, flatgrad, zipsame, get_params_flat, setFromFlat
from keras.optimizers import SGD
# from keras.utils.np_utils import to_categoricalnetwork
import keras.backend as K
import keras
from keras.layers import RepeatVector
from keras.models import Sequential, Model
from algorithm.SiameseNetwork import *
from util.SimulationUtil import createForwardDynamicsNetwork
from keras.losses import mse, binary_crossentropy

import functools
import inspect

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.util import nest

tfd = tfp.distributions


def euclidean_distance_fd2(vects):
    x, y = vects
    return K.sum(K.square(x - y), axis=-1, keepdims=True)

def l1_distance_fd2(vects):
    x, y = vects
    return K.sum(K.abs(x - y), axis=-1, keepdims=True)

def eucl_dist_output_shape_fd2(shapes):
    shape1, shape2 = shapes
    return (shape1[0], shape1[1], 1)


def vae_loss(network_vae, network_vae_log_var):
    def _vae_loss(y_true, y_pred):
        reconstruction_loss_a = mse(y_true, y_pred)
        reconstruction_loss_a *= 4096
        kl_loss = 1 + network_vae_log_var - K.square(network_vae) - K.exp(network_vae_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss_a + kl_loss)
        return vae_loss

# reparameterization trick from Keras example
# https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def map_distribution_structure(func, *dist_structure):
  def _get_params(dist):
    return {k: v for k, v in dist.parameters.items() if isinstance(v, tf.Tensor)}

  def _get_other_params(dist):
    return {k: v for k, v in dist.parameters.items() if not isinstance(v, tf.Tensor)}

  def _func(*dist_list):
    # all dists should be instances of the same class
    for dist in dist_list[1:]:
      assert dist.__class__ == dist_list[0].__class__
    dist_ctor = dist_list[0].__class__

    dist_other_params_list = [_get_other_params(dist) for dist in dist_list]

    # all dists should have the same non-tensor params
    for dist_other_params in dist_other_params_list[1:]:
      assert dist_other_params == dist_other_params_list[0]
    dist_other_params = dist_other_params_list[0]

    # filter out params that are not in the constructor's signature
    sig = inspect.signature(dist_ctor)
    dist_other_params = {k: v for k, v in dist_other_params.items() if k in sig.parameters}

    dist_params_list = [_get_params(dist) for dist in dist_list]
    values_list = [list(params.values()) for params in dist_params_list]
    values_list = list(zip(*values_list))

    structure_list = [func(*values) for values in values_list]

    values_list = [nest.flatten(structure) for structure in structure_list]
    values_list = list(zip(*values_list))
    dist_params_list = [dict(zip(dist_params_list[0].keys(), values)) for values in values_list]
    dist_list = [dist_ctor(**params, **dist_other_params) for params in dist_params_list]

    dist_structure = nest.pack_sequence_as(structure_list[0], dist_list)
    return dist_structure

  return nest.map_structure(_func, *dist_structure)


class StepType(object):
  """Defines the status of a `TimeStep` within a sequence."""
  # Denotes the first `TimeStep` in a sequence.
  FIRST = np.asarray(0, dtype=np.int32)
  # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
  MID = np.asarray(1, dtype=np.int32)
  # Denotes the last `TimeStep` in a sequence.
  LAST = np.asarray(2, dtype=np.int32)

  def __new__(cls, value):
    """Add ability to create StepType constants from a value."""
    if value == cls.FIRST:
      return cls.FIRST
    if value == cls.MID:
      return cls.MID
    if value == cls.LAST:
      return cls.LAST

    raise ValueError('No known conversion for `%r` into a StepType' % value)


class Bernoulli(tf.keras.Model):
  def __init__(self, base_depth):
    super(Bernoulli, self).__init__()
    self.dense1 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.dense2 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.output_layer = tf.keras.layers.Dense(1)

  def call(self, *inputs):
    if len(inputs) > 1:
      inputs = tf.concat(inputs, axis=-1)
    else:
      inputs, = inputs
    out = self.dense1(inputs)
    out = self.dense2(out)
    out = self.output_layer(out)
    logits = tf.squeeze(out, axis=-1)
    return tfd.Bernoulli(logits=logits)


class Normal(tf.keras.Model):
  def __init__(self, base_depth, scale):
    super(Normal, self).__init__()
    self.scale = scale
    self.dense1 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.dense2 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.output_layer = tf.keras.layers.Dense(2 if self.scale is None else 1)

  def call(self, *inputs):
    if len(inputs) > 1:
      inputs = tf.concat(inputs, axis=-1)
    else:
      inputs, = inputs
    out = self.dense1(inputs)
    out = self.dense2(out)
    out = self.output_layer(out)
    loc = out[..., 0]
    if self.scale is None:
      assert out.shape[-1].value == 2
      scale = tf.nn.softplus(out[..., 1]) + 1e-5
    else:
      assert out.shape[-1].value == 1
      scale = self.scale
    return tfd.Normal(loc=loc, scale=scale)


class MultivariateNormalDiag(tf.keras.Model):
  def __init__(self, base_depth, latent_size):
    super(MultivariateNormalDiag, self).__init__()
    self.latent_size = latent_size
    self.dense1 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.dense2 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.output_layer = tf.keras.layers.Dense(2 * latent_size)

  def call(self, *inputs):
    if len(inputs) > 1:
      inputs = tf.concat(inputs, axis=-1)
    else:
      inputs, = inputs
    out = self.dense1(inputs)
    out = self.dense2(out)
    out = self.output_layer(out)
    loc = out[..., :self.latent_size]
    scale_diag = tf.nn.softplus(out[..., self.latent_size:]) + 1e-5
    return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)


class Deterministic(tf.keras.Model):
  def __init__(self, base_depth, latent_size):
    super(Deterministic, self).__init__()
    self.latent_size = latent_size
    self.dense1 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.dense2 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.output_layer = tf.keras.layers.Dense(latent_size)

  def call(self, *inputs):
    if len(inputs) > 1:
      inputs = tf.concat(inputs, axis=-1)
    else:
      inputs, = inputs
    out = self.dense1(inputs)
    out = self.dense2(out)
    loc = self.output_layer(out)
    return tfd.Deterministic(loc=loc)


class ConstantMultivariateNormalDiag(tf.keras.Model):
  def __init__(self, latent_size):
    super(ConstantMultivariateNormalDiag, self).__init__()
    self.latent_size = latent_size

  def call(self, *inputs):
    # first input should not have any dimensions after the batch_shape, step_type
    batch_shape = tf.shape(inputs[0])  # input is only used to infer batch_shape
    shape = tf.concat([batch_shape, [self.latent_size]], axis=0)
    loc = tf.zeros(shape)
    scale_diag = tf.ones(shape)
    return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)


class ConstantDeterministic(tf.keras.Model):
  def __init__(self, latent_size):
    super(ConstantDeterministic, self).__init__()
    self.latent_size = latent_size

  def call(self, *inputs):
    # first input should not have any dimensions after the batch_shape, step_type
    batch_shape = tf.shape(inputs[0])  # input is only used to infer batch_shape
    shape = tf.concat([batch_shape, [self.latent_size]], axis=0)
    loc = tf.zeros(shape)
    return tfd.Deterministic(loc=loc)


class Decoder(tf.keras.Model):
  """Probabilistic decoder for `p(x_t | z_t)`.
  """

  def __init__(self, base_depth, channels=3, scale=1.0):
    super(Decoder, self).__init__()
    self.scale = scale
    conv_transpose = functools.partial(
        tf.keras.layers.Conv2DTranspose, padding="SAME", activation=tf.nn.leaky_relu)
    self.conv_transpose1 = conv_transpose(8 * base_depth, 4, padding="VALID")
    self.conv_transpose2 = conv_transpose(4 * base_depth, 3, 2)
    self.conv_transpose3 = conv_transpose(2 * base_depth, 3, 2)
    self.conv_transpose4 = conv_transpose(base_depth, 3, 2)
    self.conv_transpose5 = conv_transpose(channels, 5, 2)  # , activation=tf.nn.sigmoid)

  def call(self, *inputs):
    # import ipdb;ipdb.set_trace()
    if len(inputs) > 1:
      latent = tf.concat(inputs, axis=-1)
    else:
      latent, = inputs
    # (sample, N, T, latent)
    collapsed_shape = tf.stack([-1, 1, 1, tf.shape(latent)[-1]], axis=0)
    out = tf.reshape(latent, collapsed_shape)
    out = self.conv_transpose1(out)
    out = self.conv_transpose2(out)
    out = self.conv_transpose3(out)
    out = self.conv_transpose4(out)
    out = self.conv_transpose5(out)  # (sample*N*T, h, w, c)

    expanded_shape = tf.concat(
        [tf.shape(latent)[:-1], tf.shape(out)[1:]], axis=0)
    out = tf.reshape(out, expanded_shape)  # (sample, N, T, h, w, c)
    return tfd.Independent(
        distribution=tfd.Normal(loc=out, scale=self.scale),
        reinterpreted_batch_ndims=3)  # wrap (h, w, c)


class Compressor(tf.keras.Model):
  """Feature extractor.
  """

  def __init__(self, base_depth, feature_size):
    super(Compressor, self).__init__()
    self.feature_size = feature_size
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation=tf.nn.leaky_relu)
    self.conv1 = conv(base_depth, 5, 2)
    self.conv2 = conv(2 * base_depth, 3, 2)
    self.conv3 = conv(4 * base_depth, 3, 2)
    self.conv4 = conv(8 * base_depth, 3, 2)
    self.conv5 = conv(8 * base_depth, 4, padding="VALID")

  def call(self, image):
    image_shape = tf.shape(image)[-3:]
    collapsed_shape = tf.concat(([-1], image_shape), axis=0)
    out = tf.reshape(image, collapsed_shape)  # (sample*N*T, h, w, c)
    out = self.conv1(out)
    out = self.conv2(out)
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.conv5(out)
    expanded_shape = tf.concat((tf.shape(image)[:-3], [self.feature_size]), axis=0)
    return tf.reshape(out, expanded_shape)  # (sample, N, T, feature)

class SLACModel(SiameseNetwork):
    
    def __init__(self, model, state_length, action_length, state_bounds, action_bounds, settings_, reward_bounds=0, print_info=False):

        super(SLACModel,self).__init__(model, state_length, action_length, state_bounds, action_bounds, reward_bounds, settings_)
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.base_depth = base_depth
        self.latent1_size = latent1_size
        self.latent2_size = latent2_size
        self.kl_analytic = kl_analytic
        self.latent1_deterministic = latent1_deterministic
        self.latent2_deterministic = latent2_deterministic
        self.model_reward = model_reward
        self.model_discount = model_discount
        self.fps = fps
    
        if self.latent1_deterministic:
          latent1_first_prior_distribution_ctor = ConstantDeterministic
          latent1_distribution_ctor = Deterministic
        else:
          latent1_first_prior_distribution_ctor = ConstantMultivariateNormalDiag
          latent1_distribution_ctor = MultivariateNormalDiag
        if self.latent2_deterministic:
          latent2_distribution_ctor = Deterministic
        else:
          latent2_distribution_ctor = MultivariateNormalDiag
    
        # p(z_1^1)
        self.latent1_first_prior = latent1_first_prior_distribution_ctor(latent1_size)
        # p(z_1^2 | z_1^1)
        self.latent2_first_prior = latent2_distribution_ctor(8 * base_depth, latent2_size)
        # p(z_{t+1}^1 | z_t^2, a_t)
        self.latent1_prior = latent1_distribution_ctor(8 * base_depth, latent1_size)
        # p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
        self.latent2_prior = latent2_distribution_ctor(8 * base_depth, latent2_size)
    
        # q(z_1^1 | x_1)
        self.latent1_first_posterior = latent1_distribution_ctor(8 * base_depth, latent1_size)
        # q(z_1^2 | z_1^1) = p(z_1^2 | z_1^1)
        self.latent2_first_posterior = self.latent2_first_prior
        # q(z_{t+1}^1 | x_{t+1}, z_t^2, a_t)
        self.latent1_posterior = self.latent1_first_posterior
        # self.latent1_posterior = latent1_distribution_ctor(8 * base_depth, latent1_size)
        # q(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t) = p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
        # self.latent2_posterior = self.latent2_prior
        self.latent2_posterior = self.latent2_first_posterior
     
        # compresses x_t into a vector
        self.compressor = Compressor(base_depth, 8 * base_depth)
        # p(x_t | z_t^1, z_t^2)
        self.decoder = Decoder(base_depth, scale=decoder_stddev)
    
        if self.model_reward:
          # p(r_t | z_t^1, z_t^2, a_t, z_{t+1}^1, z_{t+1}^2)
          self.reward_predictor = Normal(8 * base_depth, scale=reward_stddev)
        else:
          self.reward_predictor = None
        if self.model_discount:
          # p(d_t | z_{t+1}^1, z_{t+1}^2)
          self.discount_predictor = Bernoulli(8 * base_depth)
        else:
          self.discount_predictor = None
          
        SLACModel.compile(self)
        
    @property
    def state_size(self):
        return self.latent1_size + self.latent2_size
    
    def compute_loss(self, images, actions, step_types, rewards=None, discounts=None, latent_posterior_samples_and_dists=None):
        sequence_length = step_types.shape[1].value - 1
    
        if latent_posterior_samples_and_dists is None:
          latent_posterior_samples_and_dists = self.sample_posterior(images, actions, step_types)
        (latent1_posterior_samples, latent2_posterior_samples), (latent1_posterior_dists, latent2_posterior_dists) = (
            latent_posterior_samples_and_dists)
        (latent1_conditional_prior_samples, latent2_conditional_prior_samples), _ = self.sample_conditional_prior(images[:, 0], actions, step_types)  # for visualization
        (latent1_prior_samples, latent2_prior_samples), _ = self.sample_prior(actions, step_types)  # for visualization
    
        def where_and_concat(reset_masks, first_prior_tensors, after_first_prior_tensors):
          after_first_prior_tensors = tf.where(reset_masks[:, 1:], first_prior_tensors[:, 1:], after_first_prior_tensors)
          prior_tensors = tf.concat([first_prior_tensors[:, 0:1], after_first_prior_tensors], axis=1)
          return prior_tensors
    
        reset_masks = tf.concat([tf.ones_like(step_types[:, 0:1], dtype=tf.bool),
                                 tf.equal(step_types[:, 1:], StepType.FIRST)], axis=1)
    
        latent1_reset_masks = tf.tile(reset_masks[:, :, None], [1, 1, self.latent1_size])
        latent1_first_prior_dists = self.latent1_first_prior(step_types)
        # these distributions start at t=1 and the inputs are from t-1
        latent1_after_first_prior_dists = self.latent1_prior(
            latent2_posterior_samples[:, :sequence_length], actions[:, :sequence_length])
        latent1_prior_dists = map_distribution_structure(
            functools.partial(where_and_concat, latent1_reset_masks),
            latent1_first_prior_dists,
            latent1_after_first_prior_dists)
    
        latent2_reset_masks = tf.tile(reset_masks[:, :, None], [1, 1, self.latent2_size])
        latent2_first_prior_dists = self.latent2_first_prior(latent1_posterior_samples)
        # these distributions start at t=1 and the last 2 inputs are from t-1
        latent2_after_first_prior_dists = self.latent2_prior(
            latent1_posterior_samples[:, 1:sequence_length+1],
            latent2_posterior_samples[:, :sequence_length],
            actions[:, :sequence_length])
        latent2_prior_dists = map_distribution_structure(
            functools.partial(where_and_concat, latent2_reset_masks),
            latent2_first_prior_dists,
            latent2_after_first_prior_dists)
    
        outputs = {}
    
        if self.latent1_deterministic:
          latent1_kl_divergences = 0.0
        else:
          if self.kl_analytic:
            latent1_kl_divergences = tfd.kl_divergence(latent1_posterior_dists, latent1_prior_dists)
          else:
            latent1_kl_divergences = (latent1_posterior_dists.log_prob(latent1_posterior_samples)
                                      - latent1_prior_dists.log_prob(latent1_posterior_samples))
          outputs.update({
            'latent1_kl_divergence': tf.reduce_mean(latent1_kl_divergences),
          })
        if self.latent2_deterministic:
          latent2_kl_divergences = 0.0
        else:
          if self.latent2_posterior == self.latent2_prior:
            latent2_kl_divergences = 0.0
          else:
            if self.kl_analytic:
              latent2_kl_divergences = tfd.kl_divergence(latent2_posterior_dists, latent2_prior_dists)
            else:
              latent2_kl_divergences = (latent2_posterior_dists.log_prob(latent2_posterior_samples)
                                        - latent2_prior_dists.log_prob(latent2_posterior_samples))
          outputs.update({
            'latent2_kl_divergence': tf.reduce_mean(latent2_kl_divergences),
          })
        if not self.latent1_deterministic or not self.latent2_deterministic:
          outputs.update({
            'kl_divergence': tf.reduce_mean(latent1_kl_divergences + latent2_kl_divergences),
          })
    
        likelihood_dists = self.decoder(latent1_posterior_samples, latent2_posterior_samples)
        likelihood_log_probs = likelihood_dists.log_prob(images)
        reconstruction_error = tf.reduce_sum(tf.square(images - likelihood_dists.distribution.loc),
                                             axis=list(range(-likelihood_dists.reinterpreted_batch_ndims, 0)))
        outputs.update({
          'neg_log_likelihood': -tf.reduce_mean(likelihood_log_probs),
          'reconstruction_error': tf.reduce_mean(reconstruction_error),
        })
    
        # summed over the time dimension
        elbo = tf.reduce_sum(likelihood_log_probs - latent1_kl_divergences - latent2_kl_divergences, axis=1)
    
        if self.model_reward:
          reward_dists = self.reward_predictor(
              latent1_posterior_samples[:, :sequence_length],
              latent2_posterior_samples[:, :sequence_length],
              actions[:, :sequence_length],
              latent1_posterior_samples[:, 1:sequence_length + 1],
              latent2_posterior_samples[:, 1:sequence_length + 1])
          reward_log_probs = reward_dists.log_prob(rewards[:, :sequence_length])
          reward_reconstruction_error = tf.square(rewards[:, :sequence_length] - reward_dists.loc)
          reward_valid_mask = tf.cast(tf.not_equal(step_types[:, :sequence_length], StepType.LAST), tf.float32)
          reward_log_probs *= reward_valid_mask
          reward_reconstruction_error *= reward_valid_mask
          outputs.update({
            'reward_neg_log_likelihood': -tf.reduce_mean(reward_log_probs),
            'reward_reconstruction_error': tf.reduce_mean(reward_reconstruction_error),
          })
          elbo += tf.reduce_sum(reward_log_probs, axis=1)
    
        if self.model_discount:
          discount_dists = self.discount_predictor(
              latent1_posterior_samples[:, 1:sequence_length + 1],
              latent2_posterior_samples[:, 1:sequence_length + 1])
          discount_log_probs = discount_dists.log_prob(discounts[:, :sequence_length])
          discount_accuracy = tf.cast(
              tf.equal(tf.cast(discount_dists.mode(), tf.float32), discounts[:, :sequence_length]), tf.float32)
          outputs.update({
            'discount_neg_log_likelihood': -tf.reduce_mean(discount_log_probs),
            'discount_accuracy': tf.reduce_mean(discount_accuracy),
          })
          elbo += tf.reduce_sum(discount_log_probs, axis=1)
    
        # average over the batch dimension
        elbo = tf.reduce_mean(elbo)
        loss = -elbo
    
        posterior_images = likelihood_dists.mean()
        conditional_prior_images = self.decoder(latent1_conditional_prior_samples, latent2_conditional_prior_samples).mean()
        prior_images = self.decoder(latent1_prior_samples, latent2_prior_samples).mean()
    
        outputs.update({
          'elbo': elbo,
          'images': images,
          'posterior_images': posterior_images,
          'conditional_prior_images': conditional_prior_images,
          'prior_images': prior_images,
        })
        return loss, outputs
    
    def sample_prior(self, actions, step_types=None):
        if step_types is None:
          batch_size = tf.shape(actions)[0]
          sequence_length = actions.shape[1].value  # should be statically defined
          step_types = tf.fill(
              [batch_size, sequence_length + 1], StepType.MID)
        else:
          sequence_length = step_types.shape[1].value - 1
          actions = actions[:, :sequence_length]
        
        # swap batch and time axes
        actions = tf.transpose(actions, [1, 0, 2])
        step_types = tf.transpose(step_types, [1, 0])
        
        latent1_dists = []
        latent1_samples = []
        latent2_dists = []
        latent2_samples = []
        for t in range(sequence_length + 1):
          if t == 0:
            latent1_dist = self.latent1_first_prior(step_types[t])  # step_types is only used to infer batch_size
            latent1_sample = latent1_dist.sample()
            latent2_dist = self.latent2_first_prior(latent1_sample)
            latent2_sample = latent2_dist.sample()
          else:
            reset_mask = tf.equal(step_types[t], StepType.FIRST)
            latent1_first_dist = self.latent1_first_prior(step_types[t])
            latent1_dist = self.latent1_prior(latent2_samples[t-1], actions[t-1])
            latent1_dist = map_distribution_structure(
                functools.partial(tf.where, reset_mask), latent1_first_dist, latent1_dist)
            latent1_sample = latent1_dist.sample()
        
            latent2_first_dist = self.latent2_first_prior(latent1_sample)
            latent2_dist = self.latent2_prior(latent1_sample, latent2_samples[t-1], actions[t-1])
        
            latent2_dist = map_distribution_structure(
                functools.partial(tf.where, reset_mask), latent2_first_dist, latent2_dist)
            latent2_sample = latent2_dist.sample()
        
          latent1_dists.append(latent1_dist)
          latent1_samples.append(latent1_sample)
          latent2_dists.append(latent2_dist)
          latent2_samples.append(latent2_sample)
        
        latent1_dists = map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent1_dists)
        latent1_samples = tf.stack(latent1_samples, axis=1)
        latent2_dists = map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent2_dists)
        latent2_samples = tf.stack(latent2_samples, axis=1)
        return (latent1_samples, latent2_samples), (latent1_dists, latent2_dists)

    def sample_conditional_prior(self, first_image, actions, step_types=None):
        # import ipdb ; ipdb.set_trace()
        if step_types is None:
          batch_size = tf.shape(actions)[0]
          sequence_length = actions.shape[1].value  # should be statically defined
          step_types = tf.fill(
              [batch_size, sequence_length + 1], StepType.MID)
        else:
          sequence_length = step_types.shape[1].value - 1
          actions = actions[:, :sequence_length]
        
        first_feature = self.compressor(first_image)
        
        # swap batch and time axes
        actions = tf.transpose(actions, [1, 0, 2])
        step_types = tf.transpose(step_types, [1, 0])
        
        latent1_dists = []
        latent1_samples = []
        latent2_dists = []
        latent2_samples = []
        for t in range(sequence_length + 1):
          if t == 0:
            latent1_dist = self.latent1_first_posterior(first_feature)
            latent1_sample = latent1_dist.sample()
            latent2_dist = self.latent2_first_posterior(latent1_sample)
            latent2_sample = latent2_dist.sample()
          else:
            reset_mask = tf.equal(step_types[t], StepType.FIRST)
            latent1_first_dist = self.latent1_first_prior(step_types[t])
        
            latent1_dist = self.latent1_prior(latent2_samples[t-1], actions[t-1])
            latent1_dist = map_distribution_structure(
                functools.partial(tf.where, reset_mask), latent1_first_dist, latent1_dist)
            latent1_sample = latent1_dist.sample()
        
            latent2_first_dist = self.latent2_first_prior(latent1_sample)
            latent2_dist = self.latent2_prior(latent1_sample, latent2_samples[t-1], actions[t-1])
        
            latent2_dist = map_distribution_structure(
                functools.partial(tf.where, reset_mask), latent2_first_dist, latent2_dist)
            latent2_sample = latent2_dist.sample()
        
          latent1_dists.append(latent1_dist)
          latent1_samples.append(latent1_sample)
          latent2_dists.append(latent2_dist)
          latent2_samples.append(latent2_sample)
        
        latent1_dists = map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent1_dists)
        latent1_samples = tf.stack(latent1_samples, axis=1)
        latent2_dists = map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent2_dists)
        latent2_samples = tf.stack(latent2_samples, axis=1)
        return (latent1_samples, latent2_samples), (latent1_dists, latent2_dists)

    def sample_posterior(self, images, actions, step_types, features=None):
        sequence_length = step_types.shape[1].value - 1
        actions = actions[:, :sequence_length]
        
        if features is None:
          features = self.compressor(images)
        
        # swap batch and time axes
        features = tf.transpose(features, [1, 0, 2])
        actions = tf.transpose(actions, [1, 0, 2])
        step_types = tf.transpose(step_types, [1, 0])
        
        latent1_dists = []
        latent1_samples = []
        latent2_dists = []
        latent2_samples = []
        for t in range(sequence_length + 1):
          if t == 0:
            latent1_dist = self.latent1_first_posterior(features[t])
            latent1_sample = latent1_dist.sample()
            latent2_dist = self.latent2_first_posterior(latent1_sample)
            latent2_sample = latent2_dist.sample()
          else:
            reset_mask = tf.equal(step_types[t], StepType.FIRST)
            latent1_first_dist = self.latent1_first_posterior(features[t])
            latent1_dist = self.latent1_posterior(features[t], latent2_samples[t-1], actions[t-1])
            # latent1_dist = self.latent1_posterior(features[t])
            latent1_dist = map_distribution_structure(
                functools.partial(tf.where, reset_mask), latent1_first_dist, latent1_dist)
            latent1_sample = latent1_dist.sample()
        
            latent2_first_dist = self.latent2_first_posterior(latent1_sample)
            latent2_dist = self.latent2_posterior(latent1_sample, latent2_samples[t-1], actions[t-1])
            #latent2_dist = self.latent2_posterior(latent1_sample)
            latent2_dist = map_distribution_structure(
                functools.partial(tf.where, reset_mask), latent2_first_dist, latent2_dist)
            latent2_sample = latent2_dist.sample()
        
          latent1_dists.append(latent1_dist)
          latent1_samples.append(latent1_sample)
          latent2_dists.append(latent2_dist)
          latent2_samples.append(latent2_sample)
        
        latent1_dists = map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent1_dists)
        latent1_samples = tf.stack(latent1_samples, axis=1)
        latent2_dists = map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent2_dists)
        latent2_samples = tf.stack(latent2_samples, axis=1)
        return (latent1_samples, latent2_samples), (latent1_dists, latent2_dists)
    
    def compile(self):
        # sgd = SGD(lr=0.001, momentum=0.9)
        
        # because we re-use the same instance `base_network`,
        # the weights of the network
        # will be shared across the two branches
        
        
        state_copy = keras.layers.Input(shape=keras.backend.int_shape(self._model.getStateSymbolicVariable())[1:], name="State_2")
        print ("*** self._model.getStateSymbolicVariable() shape: ", repr(keras.backend.int_shape(self._model.getStateSymbolicVariable())))
        print ("*** self._model.getStateSymbolicVariable() shape: ", repr(self._model.getStateSymbolicVariable()))

        ### Compressor
        ### outputs a multi variate diagonal normal distribution
        ### p(e|x)  
        processed_a = self._model._forward_dynamics_net(self._model.getStateSymbolicVariable())[0]
        self._model.processed_a = Model(inputs=[self._model.getStateSymbolicVariable()], outputs=processed_a, name="forward_encoder_outputs_mean")
        processed_a_log_var = self._model._forward_dynamics_net(self._model.getStateSymbolicVariable())[1]
        self._model.processed_a_log_var = Model(inputs=[self._model.getStateSymbolicVariable()], outputs=processed_a_log_var, name="forward_encoder_outputs_log_var")
        ### Marginal encoder
        ### p(z|x)
        processed_a_vae = self._model._forward_dynamics_net(self._model.getStateSymbolicVariable())[2]
        self._model.processed_a_vae = Model(inputs=[self._model.getStateSymbolicVariable()], outputs=processed_a_vae, name="forward_encoder_outputs_z")
        ### Distribute compressor over sequence
        ### p(mu_0, ... , mu_t|x_0, ..., x_t)
        network_ = keras.layers.TimeDistributed(self._model.processed_a, input_shape=(None, 1, self._state_length), name='forward_mean_encoding')(self._model.getResultStateSymbolicVariable())
        print ("network_: ", repr(network_))
        ### p(sig_0, ... , sig_t|x_0, ..., x_t)
        self._network_vae_log_var = keras.layers.TimeDistributed(self._model.processed_a_log_var, input_shape=(None, 1, self._state_length), name='forward_log_var')(self._model.getResultStateSymbolicVariable())
        print ("network_vae: ", repr(network_))
        ### This will be used later mostly as an auxilerary loss, maybe will be the marginal z model at some point.
        ### p(z_0, ... , z_t|x_0, ..., x_t)
        self._network_vae = keras.layers.TimeDistributed(self._model.processed_a_vae, input_shape=(None, 1, self._state_length), name='forward_z_sample_seq')(self._model.getResultStateSymbolicVariable())
        print ("network_vae: ", repr(network_))
        
        ### p(z_1^1)
        batch = K.shape(self._model.getStateSymbolicVariable())[0]
        dim = self.getSettings()["encoding_vector_size"]
        self.latent1_first_prior = K.random_normal(shape=(batch, dim))
        # p(z_1^2 | z_1^1)
        self.latent2_first_prior = self._model._reward_net([0, self.latent1_first_prior, 0])
        
        # p(z_{t+1}^1 | z_t^2, a_t)
        self.latent1_prior = self._modelTarget._reward_net(self.latent2_first_prior, self._model._Action)

        ### # p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
        self.latent2_prior  = self._model._reward_net([self.latent1_prior, self.latent1_first_prior, self._model._Action])
        
        # q(z_1^1 | x_1)
        # self.latent1_first_posterior = latent1_distribution_ctor(8 * base_depth, latent1_size)
        self.latent1_first_posterior = self._modelTarget._reward_net(self.latent2_first_prior, 0)
        # q(z_1^2 | z_1^1) = p(z_1^2 | z_1^1)
        self.latent2_first_posterior = self.latent2_first_prior
        # q(z_{t+1}^1 | x_{t+1}, z_t^2, a_t)
        self.latent1_posterior = self.latent1_first_posterior
        # self.latent1_posterior = latent1_distribution_ctor(8 * base_depth, latent1_size)
        # q(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t) = p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
        # self.latent2_posterior = self.latent2_prior
        self.latent2_posterior = self.latent2_first_posterior
        
        encode_input__ = keras.layers.Input(shape=keras.backend.int_shape(state_h)[1:]
                                                                          , name="seq_encoding_input"
                                                                          )
        self.seq_mean = keras.layers.Dense(self.getSettings()["encoding_vector_size"], activation = 'linear', name="seq_mean")(encode_input__)
        self._seq_mean = Model(inputs=[encode_input__], outputs=self.seq_mean, name="seq_mean")
        self.seq_log_var = keras.layers.Dense(self.getSettings()["encoding_vector_size"], activation = 'sigmoid',  name="seq_log_var")(encode_input__)
        self._seq_log_var = Model(inputs=[encode_input__], outputs=self.seq_log_var, name="seq_log_var")
        
        self._seq_z_mean = self._seq_mean(encode_input__)
        self._seq_z_log_var = self._seq_log_var(encode_input__)
        self._seq_z = keras.layers.Lambda(sampling, output_shape=(self.getSettings()["encoding_vector_size"],), name='seq_z_sampling')([self._seq_z_mean, 
                                                                   self._seq_z_log_var])
        self._seq_z = Model(inputs=[encode_input__], outputs=self._seq_z, name='seq_z_sampling')
        
        self._seq_mean = keras.layers.TimeDistributed(self._seq_mean, input_shape=(None, 1, 67), name="after_lsmt_seq_mean" )(lstm_seq)
        self._seq_log_var = keras.layers.TimeDistributed(self._seq_log_var, input_shape=(None, 1, 67), name="after_lsmt_seq_log_var")(lstm_seq)
        self._seq_z_seq = keras.layers.TimeDistributed(self._seq_z, input_shape=(None, 1, 67), name="after_lsmt_seq_z")(lstm_seq)
        
        self.seq_mean_2 = keras.layers.Dense(self.getSettings()["encoding_vector_size"], activation = 'linear', name="seq_mean")(encode_input__)
        self._seq_mean_2 = Model(inputs=[encode_input__], outputs=self.seq_mean_2, name="seq_mean")
        self.seq_log_var_2 = keras.layers.Dense(self.getSettings()["encoding_vector_size"], activation = 'sigmoid',  name="seq_log_var")(encode_input__)
        self._seq_log_var_2 = Model(inputs=[encode_input__], outputs=self.seq_log_var_2, name="seq_log_var")
        
        self._seq_z_mean_2 = self._seq_mean_2(encode_input__)
        self._seq_z_log_var_2 = self._seq_log_var_2(encode_input__)
        self._seq_z_2 = keras.layers.Lambda(sampling, output_shape=(self.getSettings()["encoding_vector_size"],), name='seq_z_2_sampling')([self._seq_z_mean_2, 
                                                                   self._seq_z_log_var_2])
        self._seq_z_2 = Model(inputs=[encode_input__], outputs=self._seq_z_2, name='seq_z_2_sampling')
        
        self._seq_mean_2 = keras.layers.TimeDistributed(self._seq_mean_2, input_shape=(None, 1, 67), name="after_lsmt_seq_mean_2" )(lstm_seq_2)
        self._seq_log_var_2 = keras.layers.TimeDistributed(self._seq_log_var_2, input_shape=(None, 1, 67), name="after_lsmt_seq_log_var_2")(lstm_seq_2)
        self._seq_z_seq_2 = keras.layers.TimeDistributed(self._seq_z_2, input_shape=(None, 1, 67), name="after_lsmt_seq_z_2")(lstm_seq_2)
        # self._model.processed_a_r = Model(inputs=[self._model.getResultStateSymbolicVariable()], outputs=self._seq_mean)
        # self._model.processed_b_r = Model(inputs=[result_state_copy], outputs=processed_b_r[0])
        
        ### Decode sequences into images
        # state_copy = keras.layers.Input(shape=keras.backend.int_shape(self._model.getStateSymbolicVariable())[1:], name="State_2")
        decode_seq_vae = keras.layers.TimeDistributed(self._modelTarget._forward_dynamics_net, input_shape=(None, 1, 67), name="decode_conditional_seq_z")(self._seq_z_seq)
        print ("decode_seq_vae: ", repr(decode_seq_vae))
        ### This is not really the same as the marginal over the conditional z's...
        decode_marginal_vae = keras.layers.TimeDistributed(self._modelTarget._forward_dynamics_net, input_shape=(None, 1, 67), name="decode_marginal_seq_z")(self._network_vae)
        print ("decode_marginal_vae: ", repr(decode_marginal_vae))

#         self._model._forward_dynamics_net = Model(inputs=[self._model.getStateSymbolicVariable()
#                                                           ]
#                                                   , outputs=distance_fd
#                                                   )
        
        self._model._reward_net = Model(inputs=[self._model.getResultStateSymbolicVariable(),
                                                self._model._Action
                                                      ]
                                                      , outputs=[
                                                                 decode_seq_vae, 
                                                                 decode_marginal_vae,
                                                                 ],
                                                      name='seq_vae_model'
                                                      )

        # sgd = SGD(lr=0.0005, momentum=0.9)
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), beta_1=np.float32(0.95), 
                                    beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0),
                                    clipnorm=2.5)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("sgd, actor: ", sgd)
            print ("Clipping: ", sgd.decay)
        self._model._forward_dynamics_net.compile(loss=contrastive_loss, optimizer=sgd)
        self._modelTarget._forward_dynamics_net.compile(loss=contrastive_loss, optimizer=sgd)

        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), beta_1=np.float32(0.95), 
                                    beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0),
                                    clipnorm=2.5)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("sgd, actor: ", sgd)
            print ("Clipping: ", sgd.decay)
            
            
        self._model._reward_net.compile(
                                        loss=[self.vae_seq_loss
                                             ,self.vae_marginal_
                                              ], 
                                        optimizer=sgd
                                        ,loss_weights=[0.9,0.1]
                                        )
        
        # self.reward = K.function([self._model.getStateSymbolicVariable(), self._model.getActionSymbolicVariable(), K.learning_phase()], [self._reward])
    def vae_marginal_(self, action_true, action_pred):
        
        reconstruction_loss = mse(action_true, action_pred)
        # reconstruction_loss *= 4096
        kl_loss = 1 + self._network_vae_log_var - K.square(self._network_vae) - K.exp(self._network_vae_log_var)
        ### Using mean 
        kl_loss = K.mean(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss_a = K.mean(reconstruction_loss + kl_loss)
        return vae_loss_a

    def vae_seq_loss(self, action_true, action_pred):
        
        reconstruction_loss = mse(action_true, action_pred)
        # reconstruction_loss *= 4096
        ### log p(x_t|z_t) loss
        kl_loss = 1 + self._seq_log_var - K.square(self._seq_z_seq) - K.exp(self._seq_log_var)
        ### Using mean 
        kl_loss = K.mean(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss_a = K.mean(reconstruction_loss + kl_loss)
        kl_ = kl_D_keras(self._seq_mean, self._seq_log_var, 
                         self._seq_log_var_2, self._seq_log_var_2, self.getSettings()["encoding_vector_size"])
        return vae_loss_a + kl_

    def reset(self):
        """
            Reset any state for the agent model
        """
        self._model.reset()
        self._model._reward_net.reset_states()
        self._model._forward_dynamics_net.reset_states()
        if not (self._modelTarget is None):
            self._modelTarget._forward_dynamics_net.reset_states()
            # self._modelTarget._reward_net.reset_states()
            # self._modelTarget.reset()
            
    def getNetworkParameters(self):
        params = []
        params.append(copy.deepcopy(self._model._forward_dynamics_net.get_weights()))
        params.append(copy.deepcopy(self._model._reward_net.get_weights()))
        
        if ( "return_rnn_sequence" in self.getSettings()
             and (self.getSettings()["return_rnn_sequence"])):
            params.append(copy.deepcopy(self._model._reward_net_seq.get_weights()))
                
        return params
    
    def setNetworkParameters(self, params):
        self._model._forward_dynamics_net.set_weights(params[0])
        self._model._reward_net.set_weights(params[1])
        if ( "return_rnn_sequence" in self.getSettings()
             and (self.getSettings()["return_rnn_sequence"])):
            self._model._reward_net_seq.set_weights(params[2])
        
    def setGradTarget(self, grad):
        self._fd_grad_target_shared.set_value(grad)
        
    def getGrads(self, states, actions, result_states, v_grad=None, alreadyNormed=False):
        if ( alreadyNormed == False ):
            states = np.array(norm_state(states, self.getStateBounds()), dtype=self.getSettings()['float_type'])
            actions = np.array(norm_action(actions, self._action_bounds), dtype=self.getSettings()['float_type'])
            result_states = np.array(norm_state(result_states, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        # result_states = np.array(result_states, dtype=self.getSettings()['float_type'])
        # self.setData(states, actions, result_states)
        # if (v_grad != None):
        # print ("states shape: ", states.shape, " actions shape: ", actions.shape, " v_grad.shape: ", v_grad.shape)
        self.setGradTarget(v_grad)
        # print ("states shape: ", states.shape, " actions shape: ", actions.shape)
        # grad = self._get_grad([states, actions])[0]
        grad = np.zeros_like(states)
        # print ("grad: ", grad)
        return grad
    
    def getRewardGrads(self, states, actions, alreadyNormed=False):
        # states = np.array(states, dtype=self.getSettings()['float_type'])
        # actions = np.array(actions, dtype=self.getSettings()['float_type'])
        if ( alreadyNormed is False ):
            states = np.array(norm_state(states, self.getStateBounds()), dtype=self.getSettings()['float_type'])
            actions = np.array(norm_action(actions, self._action_bounds), dtype=self.getSettings()['float_type'])
            # rewards = np.array(norm_state(rewards, self._reward_bounds), dtype=self.getSettings()['float_type'])
        # self.setData(states, actions)
        return self._get_grad_reward([states, actions, 0])[0]
    
    def updateTargetModel(self):
        pass
                
    def train(self, states, actions, result_states, rewards, falls=None, updates=1, batch_size=None, p=1, lstm=True, datas=None):
        """
            states will come for the agent and
            results_states can come from the imitation agent
        """
        # print ("fd: ", self)
        # print ("state length: ", len(self.getStateBounds()[0]))
        self.reset()
        states_ = states
        if ('anneal_learning_rate' in self.getSettings()
            and (self.getSettings()['anneal_learning_rate'] == True)):
            K.set_value(self._model._forward_dynamics_net.optimizer.lr, np.float32(self.getSettings()['fd_learning_rate']) * p)

        if (((("train_LSTM_FD" in self._settings)
                and (self._settings["train_LSTM_FD"] == True))
            or
            (("train_LSTM_Reward" in self._settings)
                and (self._settings["train_LSTM_Reward"] == True))
            ) 
            and lstm):
            ### result states can be from the imitation agent.
            # print ("falls: ", falls)
            # print ("sequences0 shape: ", sequences0.shape)
            loss_ = []
        
            # print ("targets_[:,:,0]: ", np.mean(targets_, axis=1))
            # print ("targets__: ", targets__)
            if (("train_LSTM_FD" in self._settings)
                and (self._settings["train_LSTM_FD"] == True)):
                score = self._model._forward_dynamics_net.fit([states], [states],
                              epochs=1, 
                              batch_size=sequences0.shape[0],
                              verbose=0
                              )
                loss_.append(np.mean(score.history['loss']))
                
            if (("train_LSTM_Reward" in self._settings)
                and (self._settings["train_LSTM_Reward"] == True)):
                
                print("states shape: ", states.shape)
                print("actions shape: ", actions.shape)
                score = self._model._reward_net.fit([states, actions], 
                              [states, 
                               states],
                              epochs=1, 
                              batch_size=states.shape[0],
                              verbose=0
                              )
                
                loss_.append(np.mean(score.history['loss']))
            
            return np.mean(loss_)
        else:
            te_pair1, te_pair2, te_y = create_pairs2(states_, self._settings)
        self._updates += 1
        if (batch_size is None):
            batch_size_=states.shape[0]
        else:
            batch_size_=batch_size
        loss = 0
        # dist_ = np.array(self._contrastive_loss([te_pair1, te_pair2, 0]))[0]
        # dist = np.mean(dist_)
        te_y = np.array(te_y)
        # print("Distance: ", dist)
        # print("targets: ", te_y)
        # print("pairs: ", te_pair1)
        # print("Distance.shape, targets.shape: ", dist_.shape, te_y.shape)
        # print("Distance, targets: ", np.concatenate((dist_, te_y), axis=1))
        # if ( dist > 0):
        score = self._model._forward_dynamics_net.fit([te_pair1, te_pair2], te_y,
          epochs=updates, batch_size=batch_size_,
          verbose=0,
          shuffle=True
          )
        loss = np.mean(score.history['loss'])
            # print ("loss: ", loss)
        return loss
    
    def predict_encoding(self, state):
        """
            Compute distance between two states
        """
        # state = np.array(norm_state(state, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        if (("train_LSTM_FD" in self._settings)
                    and (self._settings["train_LSTM_FD"] == True)):
            h_a = self._model.processed_a.predict([np.array([state])])
        else:
            h_a = self._model._forward_dynamics_net.predict([state])[0]
        return h_a
    
    def predict(self, state, state2):
        """
            Compute distance between two states
        """
        # print("state shape: ", np.array(state).shape)
        state = np.array(norm_state(state, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        state2 = np.array(norm_state(state2, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        if ((("train_LSTM_FD" in self._settings)
                    and (self._settings["train_LSTM_FD"] == True))
                    # or
                    # settings["use_learned_reward_function"] == "dual"
                    ):
            ### Used because we need to keep two separate RNN networks and not mix the hidden states
            h_a = self._model.processed_a.predict([np.array([state])])
            h_b = self._model.processed_b.predict([np.array([state2])])
            state_ = self._distance_func_np((h_a, h_b))[0]
            # print ("siamese dist: ", state_)
            # state_ = self._model._forward_dynamics_net.predict([np.array([state]), np.array([state2])])[0]
        else:
            state_ = self._model._forward_dynamics_net.predict([state, state2])[0]
        # dist_ = np.array(self._contrastive_loss([te_pair1, te_pair2, 0]))[0]
        # print("state_ shape: ", np.array(state_).shape)
        return state_
    
    def predictWithDropout(self, state, action):
        # "dropout"
        state = np.array(norm_state(state, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        self._model.setStates(state)
        self._model.setActions(action)
        state_ = scale_state(self._forwardDynamics_drop()[0], self.getStateBounds())
        return state_
    
    def predict_std(self, state, action, p=1.0):
        state = np.array(norm_state(state, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        state_ = self._forwardDynamics_std() * (action_bound_std(self.getStateBounds()))
        return state_
    
    def predict_reward(self, state, state2):
        """
            Predict reward which is inverse of distance metric
        """
        # print ("state bounds length: ", self.getStateBounds())
        # print ("fd: ", self)
        state = np.array(norm_state(state, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        state2 = np.array(norm_state(state2, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        if (("train_LSTM_Reward" in self._settings)
            and (self._settings["train_LSTM_Reward"] == True)):
            ### Used because we need to keep two separate RNN networks and not mix the hidden states
            # print ("State shape: ", np.array([np.array([state])]).shape)
            h_a = self._model.processed_a_r.predict([np.array([state])])
            h_b = self._model.processed_b_r.predict([np.array([state2])])
            reward_ = self._distance_func_np((h_a, h_b))[0]
            # print ("siamese dist: ", state_)
            # state_ = self._model._forward_dynamics_net.predict([np.array([state]), np.array([state2])])[0]
        else:
            predicted_reward = self._model._reward_net.predict([state, state2])[0]
            # reward_ = scale_reward(predicted_reward, self.getRewardBounds()) # * (1.0 / (1.0- self.getSettings()['discount_factor']))
            reward_ = predicted_reward
            
        return reward_
    
    def predict_reward_encoding(self, state):
        """
            Predict reward which is inverse of distance metric
        """
        # state = np.array(norm_state(state, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        if (("train_LSTM_Reward" in self._settings)
            and (self._settings["train_LSTM_Reward"] == True)):
            h_a = self._model.processed_a_r.predict([np.array([state])])
        else:
            h_a = self._model._reward_net.predict([state])[0]
            # reward_ = scale_reward(predicted_reward, self.getRewardBounds()) # * (1.0 / (1.0- self.getSettings()['discount_factor']))
            
        return h_a
    
    def predict_batch(self, states, actions):
        ## These input should already be normalized.
        return self.fd([states, actions, 0])[0]
    
    def predict_reward_batch(self, states, actions):
        """
            This data should already be normalized
        """
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        predicted_reward = self.reward([states, actions, 0])[0]
        return predicted_reward
    
    def predict_reward_(self, states, states2):
        """
            This data should already be normalized
        """
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        states = np.array(norm_state(states, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        actions = np.array(norm_state(states2, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        h_a = self._model.processed_a_r_seq.predict([states])
        h_b = self._model.processed_b_r_seq.predict([states2])
        # print ("h_b shape: ", h_b.shape) 
        predicted_reward = np.array([self._distance_func_np((np.array([h_a_]), np.array([h_b_])))[0] for h_a_, h_b_ in zip(h_a[0], h_b[0])])
        # print ("predicted_reward_: ", predicted_reward)
        # predicted_reward = self._model._reward_net_seq.predict([states, actions], batch_size=1)[0]
        return predicted_reward

    def bellman_error(self, states, actions, result_states, rewards):
        self.reset()
        if (("train_LSTM_FD" in self._settings)
                    and (self._settings["train_LSTM_FD"] == True)):
            errors=[]
       
            predicted_y = self._model._forward_dynamics_net.predict([states], batch_size=sequences0.shape[0])
            # print ("fd error, predicted_y: ", predicted_y)
            targets__ = np.mean(targets_, axis=1)
            # print ("fd error, targets_ : ", targets_)
            # print ("fd error, targets__: ", targets__)
            errors.append( compute_accuracy(predicted_y, targets__) )
            # predicted_y = self._model._forward_dynamics_net.predict([np.array([[sequences0[0]]]), np.array([[sequences1[0]]])])
            # te_acc = compute_accuracy(predicted_y, np.array([targets_[0]]) )
            te_acc = np.mean(errors)
        else:
            states = np.concatenate((states, result_states), axis=0)
            te_pair1, te_pair2, te_y = create_pairs2(states, self._settings)
        
            # state_ = self._model._forward_dynamics_net.predict([state, state2])[0]
#             predicted_y = self._model._forward_dynamics_net.predict([te_pair1, te_pair2])
            te_acc = 0
            
        # predicted_y = self._model._forward_dynamics_net.predict([te_pair1, te_pair2])
        return te_acc
    
    def reward_error(self, states, actions, result_states, rewards):
        # rewards = rewards * (1.0/(1.0-self.getSettings()['discount_factor'])) # scale rewards
        self.reset()
        if (("train_LSTM_Reward" in self._settings)
                    and (self._settings["train_LSTM_Reward"] == True)):
            errors=[]
            predicted_y = self._model._reward_net.predict([states, actions], batch_size=states.shape[0])
            errors.append( np.mean(predicted_y[0] - states ))
            # predicted_y = self._model._forward_dynamics_net.predict([np.array([[sequences0[0]]]), np.array([[sequences1[0]]])])
            # te_acc = compute_accuracy(predicted_y, np.array([targets_[0]]) )
            te_acc = np.mean(errors)
        else:
            states = np.concatenate((states, result_states), axis=0)
            te_pair1, te_pair2, te_y = create_pairs2(states, self._settings)
        
            # state_ = self._model._forward_dynamics_net.predict([state, state2])[0]
            predicted_y = self._model._reward_net.predict([te_pair1, te_pair2])
            te_acc = compute_accuracy(predicted_y, te_y)
            
        # predicted_y = self._model._forward_dynamics_net.predict([te_pair1, te_pair2])
        return te_acc

    def saveTo(self, fileName, states=None, actions=None):
        # print(self, "saving model")
        import h5py
        self.reset()
        hf = h5py.File(fileName+"_bounds.h5", "w")
        hf.create_dataset('_state_bounds', data=self.getStateBounds())
        hf.create_dataset('_reward_bounds', data=self.getRewardBounds())
        hf.create_dataset('_action_bounds', data=self.getActionBounds())
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("fd save self.getStateBounds(): ", len(self.getStateBounds()[0]))
        # hf.create_dataset('_resultgetStateBounds()', data=self.getResultStateBounds())
        # print ("fd: ", self)
        hf.flush()
        hf.close()
        suffix = ".h5"
        ### Save models
        # self._model._actor_train.save(fileName+"_actor_train"+suffix, overwrite=True)
        self._model._forward_dynamics_net.save(fileName+"_FD"+suffix, overwrite=True)
        # self._model._reward_net.save(fileName+"_reward"+suffix, overwrite=True)
        self._model._reward_net.save_weights(fileName+"_reward"+suffix, overwrite=True)
        self._modelTarget._forward_dynamics_net.save(fileName+"_FD_T"+suffix, overwrite=True)
        # self._modelTarget._forward_dynamics_net.save(fileName+"_FD_T"+suffix, overwrite=True)
        # self._model._reward_net.save(fileName+"_reward"+suffix, overwrite=True)
        # self._modelTarget._reward_net.save_weights(fileName+"_reward_T"+suffix, overwrite=True)
        # print ("self._model._actor_train: ", self._model._actor_train)
        try:
            from keras.utils import plot_model
            ### Save model design as image
            plot_model(self._model._forward_dynamics_net, to_file=fileName+"_FD"+'.svg', show_shapes=True)
            plot_model(self._model._reward_net, to_file=fileName+"_reward"+'.svg', show_shapes=True)
            plot_model(self._modelTarget._forward_dynamics_net, to_file=fileName+"_FD_decode"+'.svg', show_shapes=True)
        except Exception as inst:
            ### Maybe the needed libraries are not available
            print ("Error saving diagrams for rl models.")
            print (inst)
            
        if (states is not None):
            ## Don't use Xwindows backend for this
            import matplotlib
            # matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            # img_ = np.reshape(viewData, (150,158,3))
            ### get the sequence prediction
            predicted_y = self._model._reward_net.predict([states, actions], batch_size=states.shape[0])
            
            img_ = predicted_y[0]
            img_z = predicted_y[1]
            ### Get first sequence in batch
            img_ = img_[0]
            img_x = states[0]
            import imageio
            import PIL
            from PIL import Image
            images_x = []
            images_y = []
            images_z = []
            for i in range(len(img_)):
                img__y = np.reshape(img_[i], self._settings["fd_terrain_shape"])
                images_y.append(Image.fromarray(img__y).resize((256,256))) ### upsampling
                print("img_ shape", img__y.shape, " sum: ", np.sum(img__y))
                # fig1 = plt.figure(2)
                ### Save generated image
                # plt.imshow(img__y, origin='lower')
                # plt.title("agent visual Data: ")
                # fig1.savefig(fileName+"viz_state_"+str(i)+".png")
                ### Save input image
                img__x = np.reshape(img_x[i], self._settings["fd_terrain_shape"])
                images_x.append(Image.fromarray(img__x).resize((256,256))) ### upsampling
#                 plt.imshow(img__x, origin='lower')
#                 plt.title("agent visual Data: ")
#                 fig1.savefig(fileName+"viz_state_input_"+str(i)+".png")
#                 img__z = np.reshape(img_z[i], self._settings["fd_terrain_shape"])
#                 images_z.append(img__z)
                
            imageio.mimsave(fileName+"viz_state_input_"+'.gif', images_x, duration=0.5,)
            imageio.mimsave(fileName+"viz_conditional_"+'.gif', images_y, duration=0.5,)
#             imageio.mimsave(fileName+"viz_marginal_"+'.gif', images_z, duration=0.5,)
        
    def loadFrom(self, fileName):
        import h5py
        from util.utils import load_keras_model
        # from keras.models import load_weights
        suffix = ".h5"
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("Loading agent: ", fileName)
        # with K.get_session().graph.as_default() as g:
        ### Need to lead the model this way because the learning model's State expects batches...
        forward_dynamics_net = load_keras_model(fileName+"_FD"+suffix, custom_objects={'contrastive_loss': contrastive_loss})
        #reward_net = load_keras_model(fileName+"_reward"+suffix, custom_objects={'contrastive_loss': contrastive_loss,
        #                                                                         "vae_loss_a": self.vae_loss_a,
        #                                                                         "vae_loss_b": self.vae_loss_b})
        # if ("simulation_model" in self.getSettings() and
        #     (self.getSettings()["simulation_model"] == True)):
        if (True): ### Because the simulation and learning use different model types (statefull vs stateless lstms...)
            self._model._forward_dynamics_net.set_weights(forward_dynamics_net.get_weights())
            self._model._forward_dynamics_net.optimizer = forward_dynamics_net.optimizer
            # self._model._reward_net.set_weights(reward_net.get_weights())
            self._model._reward_net.load_weights(fileName+"_reward"+suffix)
            # self._model._reward_net.optimizer = reward_net.optimizer
        else:
            self._model._forward_dynamics_net = forward_dynamics_net
            self._model._reward_net = reward_net
            
        self._forward_dynamics_net = self._model._forward_dynamics_net
        self._reward_net = self._model._reward_net
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("******** self._forward_dynamics_net: ", self._forward_dynamics_net)
        if (self._modelTarget is not None):
            self._modelTarget._forward_dynamics_net = load_keras_model(fileName+"_FD_T"+suffix, custom_objects={'contrastive_loss': contrastive_loss})
            # self._modelTarget._reward_net = load_keras_model(fileName+"_reward_net_T"+suffix)
            # self._modelTarget._reward_net.load_weights(fileName+"_reward_T"+suffix)
        # self._model._actor_train = load_keras_model(fileName+"_actor_train"+suffix, custom_objects={'loss': pos_y})
        # self._value = K.function([self._model.getStateSymbolicVariable(), K.learning_phase()], [self.__value])
        # self._value_Target = K.function([self._model.getResultStateSymbolicVariable(), K.learning_phase()], [self.__value_Target])
        hf = h5py.File(fileName+"_bounds.h5",'r')
        self.setStateBounds(np.array(hf.get('_state_bounds')))
        self.setRewardBounds(np.array(hf.get('_reward_bounds')))
        self.setActionBounds(np.array(hf.get('_action_bounds')))
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print("fd load self.getStateBounds(): ", len(self.getStateBounds()[0]))
        # self._resultgetStateBounds() = np.array(hf.get('_resultgetStateBounds()'))
        hf.close()
        