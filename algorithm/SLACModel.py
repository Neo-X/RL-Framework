###
# python3 trainModel.py --config=settings/terrainRLImitate/PPO/SLACModel_mini.json -p 4 --bootstrap_samples=1000 --max_epoch_length=16 --rollouts=4 --skip_rollouts=true --train_actor=false --train_critic=false --epochs=32 --fd_updates_per_actor_update=64 --on_policy=fast
### Easy example to use with less requirements:
# python3 -m pdb -c c trainModel.py --config=settings/MiniGrid/TagEnv/PPO/Tag_SLAC_mini.json -p 2 --bootstrap_samples=1000 --max_epoch_length=16 --rollouts=4 --skip_rollouts=true --train_actor=false --train_critic=false --epochs=32 --fd_updates_per_actor_update=64 --on_policy=fast --print_level=debug

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

def display_gif(images, logdir, fps=10, max_outputs=8, counter=0):
    import moviepy.editor as mpy
    images = images[:max_outputs]
    images = np.clip(images, 0.0, 1.0)
    images = (images * 255.0).astype(np.uint8)
    images = np.concatenate(images, axis=-2)
    clip = mpy.ImageSequenceClip(list(images), fps=fps)
    # clip.write_videofile(logdir+str(global_counter)+".mp4", fps=fps)
    
    # import moviepy.editor as mpy
    # clip = mpy.ImageSequenceClip(images, fps=20)
    
    # video_dir = video_dir_prefix + 'BCpolicy-gripper_state2'+str(reset_arg)+'/'
    
    # if os.path.isdir(video_dir)!=True:
    #     os.makedirs(video_dir, exist_ok = True)
    clip.write_gif(logdir+str(counter)+".gif", fps=20)

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

class Bernoulli(tf.Module):
    def __init__(self, base_depth, name=None):
        super(Bernoulli, self).__init__(name=name)
        self.dense1 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
        self.output_layer = tf.keras.layers.Dense(1)

    def __call__(self, *inputs):
        if len(inputs) > 1:
            inputs = tf.concat(inputs, axis=-1)
        else:
            (inputs,) = inputs
        out = self.dense1(inputs)
        out = self.dense2(out)
        out = self.output_layer(out)
        logits = tf.squeeze(out, axis=-1)
        return tfd.Bernoulli(logits=logits)

class Normal(tf.Module):
    def __init__(self, base_depth, scale=None, name=None):
        super(Normal, self).__init__(name=name)
        self.scale = scale
        self.dense1 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
        self.output_layer = tf.keras.layers.Dense(2 if self.scale is None else 1)

    def __call__(self, *inputs):
        if len(inputs) > 1:
            inputs = tf.concat(inputs, axis=-1)
        else:
            (inputs,) = inputs
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

class MultivariateNormalDiag(tf.Module):
    def __init__(self, base_depth, latent_size, scale=None, name=None):
        super(MultivariateNormalDiag, self).__init__(name=name)
        self.latent_size = latent_size
        self.scale = scale
        self.dense1 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
        self.output_layer = tf.keras.layers.Dense(
            2 * latent_size if self.scale is None else latent_size
        )

    def __call__(self, *inputs):
        if len(inputs) > 1:
            inputs = tf.concat(inputs, axis=-1)
        else:
            (inputs,) = inputs
        out = self.dense1(inputs)
        out = self.dense2(out)
        out = self.output_layer(out)
        loc = out[..., : self.latent_size]
        if self.scale is None:
            assert out.shape[-1].value == 2 * self.latent_size
            scale_diag = tf.nn.softplus(out[..., self.latent_size :]) + 1e-5
        else:
            assert out.shape[-1].value == self.latent_size
            scale_diag = tf.ones_like(loc) * self.scale
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

class Deterministic(tf.Module):
    def __init__(self, base_depth, latent_size, name=None):
        super(Deterministic, self).__init__(name=name)
        self.latent_size = latent_size
        self.dense1 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
        self.output_layer = tf.keras.layers.Dense(latent_size)

    def __call__(self, *inputs):
        if len(inputs) > 1:
            inputs = tf.concat(inputs, axis=-1)
        else:
            (inputs,) = inputs
        out = self.dense1(inputs)
        out = self.dense2(out)
        loc = self.output_layer(out)
        return tfd.VectorDeterministic(loc=loc)

class ConstantMultivariateNormalDiag(tf.Module):
    def __init__(self, latent_size, scale=None, name=None):
        super(ConstantMultivariateNormalDiag, self).__init__(name=name)
        self.latent_size = latent_size
        self.scale = scale

    def __call__(self, *inputs):
        # first input should not have any dimensions after the batch_shape, step_type
        batch_shape = tf.shape(inputs[0])  # input is only used to infer batch_shape
        shape = tf.concat([batch_shape, [self.latent_size]], axis=0)
        loc = tf.zeros(shape)
        if self.scale is None:
            scale_diag = tf.ones(shape)
        else:
            scale_diag = tf.ones(shape) * self.scale
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)


class Decoder(tf.Module):
    """Probabilistic decoder for `p(x_t | z_t)`."""

    def __init__(self, base_depth, channels=3, scale=1.0, name=None):
        super(Decoder, self).__init__(name=name)
        self.scale = scale
        conv_transpose = functools.partial(
            tf.keras.layers.Conv2DTranspose, padding="SAME", activation=tf.nn.leaky_relu
        )
        self.conv_transpose1 = conv_transpose(8 * base_depth, 4, padding="VALID")
        self.conv_transpose2 = conv_transpose(4 * base_depth, 3, 2)
        self.conv_transpose3 = conv_transpose(2 * base_depth, 3, 2)
        self.conv_transpose4 = conv_transpose(base_depth, 3, 2)
        self.conv_transpose5 = conv_transpose(channels, 5, 2)

    def __call__(self, *inputs):
        if len(inputs) > 1:
            latent = tf.concat(inputs, axis=-1)
        else:
            (latent,) = inputs
        # (sample, N, T, latent)
        collapsed_shape = tf.stack([-1, 1, 1, tf.shape(latent)[-1]], axis=0)
        out = tf.reshape(latent, collapsed_shape)
        out = self.conv_transpose1(out)
        out = self.conv_transpose2(out)
        out = self.conv_transpose3(out)
        out = self.conv_transpose4(out)
        out = self.conv_transpose5(out)  # (sample*N*T, h, w, c)

        expanded_shape = tf.concat([tf.shape(latent)[:-1], tf.shape(out)[1:]], axis=0)
        out = tf.reshape(out, expanded_shape)  # (sample, N, T, h, w, c)
        return tfd.Independent(
            distribution=tfd.Normal(loc=out, scale=self.scale),
            reinterpreted_batch_ndims=3,
        )  # wrap (h, w, c)

class Compressor(tf.Module):
    """Feature extractor."""

    def __init__(self, base_depth, feature_size, name=None):
        super(Compressor, self).__init__(name=name)
        self.feature_size = feature_size
        conv = functools.partial(
            tf.keras.layers.Conv2D, padding="SAME", activation=tf.nn.leaky_relu
        )
        self.conv1 = conv(base_depth, 5, 2)
        self.conv2 = conv(2 * base_depth, 3, 2)
        self.conv3 = conv(4 * base_depth, 3, 2)
        self.conv4 = conv(8 * base_depth, 3, 2)
        self.conv5 = conv(8 * base_depth, 4, padding="VALID")

    def __call__(self, image):
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

class DecoderSmall(tf.Module):
    """Probabilistic decoder for `p(x_t | z_t)`."""

    def __init__(self, base_depth, channels=3, scale=1.0, name=None):
        super(DecoderSmall, self).__init__(name=name)
        self.scale = scale
        conv_transpose = functools.partial(
            tf.keras.layers.Conv2DTranspose, padding="SAME", activation=tf.nn.leaky_relu
        )
        self.conv_transpose1 = conv_transpose(8 * base_depth, 4, padding="VALID")
#         self.conv_transpose2 = conv_transpose(4 * base_depth, 3, 2)
        self.conv_transpose3 = conv_transpose(2 * base_depth, 3, 2)
#         self.conv_transpose4 = conv_transpose(base_depth, 3, 2)
        self.conv_transpose5 = conv_transpose(channels, 5, 2)

    def __call__(self, *inputs):
        if len(inputs) > 1:
            latent = tf.concat(inputs, axis=-1)
        else:
            (latent,) = inputs
        # (sample, N, T, latent)
        collapsed_shape = tf.stack([-1, 1, 1, tf.shape(latent)[-1]], axis=0)
        out = tf.reshape(latent, collapsed_shape)
        out = self.conv_transpose1(out)
#         out = self.conv_transpose2(out)
        out = self.conv_transpose3(out)
#         out = self.conv_transpose4(out)
        out = self.conv_transpose5(out)  # (sample*N*T, h, w, c)

        expanded_shape = tf.concat([tf.shape(latent)[:-1], tf.shape(out)[1:]], axis=0)
        out = tf.reshape(out, expanded_shape)  # (sample, N, T, h, w, c)
        return tfd.Independent(
            distribution=tfd.Normal(loc=out, scale=self.scale),
            reinterpreted_batch_ndims=3,
        )  # wrap (h, w, c)

class CompressorSmall(tf.Module):
    """Feature extractor."""

    def __init__(self, base_depth, feature_size, name=None):
        super(CompressorSmall, self).__init__(name=name)
        self.feature_size = feature_size
        conv = functools.partial(
            tf.keras.layers.Conv2D, padding="SAME", activation=tf.nn.leaky_relu
        )
        self.conv1 = conv(base_depth, 5, 2)
#         self.conv2 = conv(2 * base_depth, 3, 2)
        self.conv3 = conv(4 * base_depth, 3, 2)
#         self.conv4 = conv(8 * base_depth, 3, 2)
        self.conv5 = conv(8 * base_depth, 4, padding="VALID")

    def __call__(self, image):
        image_shape = tf.shape(image)[-3:]
        collapsed_shape = tf.concat(([-1], image_shape), axis=0)
        out = tf.reshape(image, collapsed_shape)  # (sample*N*T, h, w, c)
        out = self.conv1(out)
#         out = self.conv2(out)
        out = self.conv3(out)
#         out = self.conv4(out)
        out = self.conv5(out)
        expanded_shape = tf.concat((tf.shape(image)[:-3], [self.feature_size]), axis=0)
        return tf.reshape(out, expanded_shape)  # (sample, N, T, feature)

class SLACModel(SiameseNetwork):
    
    def __init__(self, model, state_length, action_length, state_bounds, action_bounds, settings_, reward_bounds=0, print_info=False):

        super(SiameseNetwork,self).__init__(model, state_length, action_length, state_bounds, action_bounds, reward_bounds, settings_)
        observation_spec = None
        action_spec = None
        base_depth=32
        latent1_size=32
        latent2_size=256
        kl_analytic=True
        latent1_deterministic=False
        latent2_deterministic=False
        model_reward=False
        model_discount=False
        fps=None
        decoder_stddev=np.sqrt(0.1, dtype=np.float32)
        reward_stddev=None
        name='SlacModelDistributionNetwork'
        compressor=None
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
    
        latent1_first_prior_distribution_ctor = ConstantMultivariateNormalDiag
        latent1_distribution_ctor = MultivariateNormalDiag
        latent2_distribution_ctor = MultivariateNormalDiag

        # p(z_1^1)
        self.latent1_first_prior = latent1_first_prior_distribution_ctor(latent1_size, name='Latent1FirstPrior')
        # p(z_1^2 | z_1^1)
        self.latent2_first_prior = latent2_distribution_ctor(
            8 * base_depth, latent2_size, name='Latent2FirstPrior'
        )
        # p(z_{t+1}^1 | z_t^2, a_t)
        self.latent1_prior = latent1_distribution_ctor(8 * base_depth, latent1_size, name='Latent1Prior')
        # p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t) (? Conceptually similar to p(z_{t+1} | z_t, a_t) ?)
        self.latent2_prior = latent2_distribution_ctor(8 * base_depth, latent2_size, name='Latent2Prior')

        # q(z_1^1 | x_1)
        self.latent1_first_posterior = latent1_distribution_ctor(
            8 * base_depth, latent1_size, name='Latent1_FirstPosterior'
        )
        # TODO ?????????????????????????????????????????????????????????????????????????????????????????????????????
        # This next line seems extremely broken? WHY?
        # q(z_1^2 | z_1^1) = p(z_1^2 | z_1^1)
        self.latent2_first_posterior = self.latent2_first_prior
        
        # q(z_{t+1}^1 | x_{t+1}, z_t^2, a_t)
        self.latent1_posterior = latent1_distribution_ctor(8 * base_depth, latent1_size, name='Latent1Posterior')

        # TODO ?????????????????????????????????????????????????????????????????????????????????????????????????????
        # This next line seems extremely broken? WHY?
        # q(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t) = p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)        
        self.latent2_posterior = self.latent2_prior

        # compresses x_t into a vector
        if compressor is None:
            if ("slac_use_small_imgs" in self.getSettings()
                and (self.getSettings()["slac_use_small_imgs"])):
                self.compressor = CompressorSmall(base_depth, 8 * base_depth)
                # p(x_t | z_t^1, z_t^2)
                self.observation_decoder = DecoderSmall(base_depth, scale=decoder_stddev)
            
            else:
                self.compressor = Compressor(base_depth, 8 * base_depth)
                # p(x_t | z_t^1, z_t^2)
                self.observation_decoder = Decoder(base_depth, scale=decoder_stddev)
        else:
            self.compressor = compressor
            # Decode to vectors! p(x_t | z_t^1, z_t^2)
            # HACK
            assert(observation_spec['pixels'].shape.rank == 1)
            observation_dimension = observation_spec['pixels'].shape[0].value
            self.observation_decoder = MultivariateNormalDiag(base_depth, observation_dimension)
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

    def compute_loss(
        self,
        images,
        actions,
        step_types,
        rewards=None,
        discounts=None,
        latent_posterior_samples_and_dists=None,
    ):
        sequence_length = step_types.shape[1].value - 1
        # semihack for discrete actions
        actions = tf.cast(actions, dtype=tf.float32)

        if latent_posterior_samples_and_dists is None:
            latent_posterior_samples_and_dists = self.sample_posterior(
                images, actions, step_types
            )
        (
            (latent1_posterior_samples, latent2_posterior_samples),
            (latent1_posterior_dists, latent2_posterior_dists),
        ) = latent_posterior_samples_and_dists

        # for visualization
        ((latent1_prior_samples, latent2_prior_samples), _) = self.sample_prior_or_posterior(actions, step_types)  

        # for visualization. condition on first image only
        ((latent1_conditional_prior_samples, latent2_conditional_prior_samples), _) = self.sample_prior_or_posterior(
            actions, step_types, images=images[:, :1])

        def where_and_concat(reset_masks, first_prior_tensors, after_first_prior_tensors):
            after_first_prior_tensors = tf.where(
                reset_masks[:, 1:],
                first_prior_tensors[:, 1:],
                after_first_prior_tensors,
            )
            prior_tensors = tf.concat(
                [first_prior_tensors[:, 0:1], after_first_prior_tensors], axis=1
            )
            return prior_tensors

        reset_masks = tf.concat([tf.ones_like(step_types[:, 0:1], dtype=tf.bool),
                                 tf.equal(step_types[:, 1:], StepType.FIRST), ], axis=1)

        latent1_reset_masks = tf.tile(reset_masks[:, :, None], [1, 1, self.latent1_size])
        
        latent1_first_prior_dists = self.latent1_first_prior(step_types)

        # these distributions start at t=1 and the inputs are from t-1
        latent1_after_first_prior_dists = self.latent1_prior(latent2_posterior_samples[:, :sequence_length],
                                                             actions[:, :sequence_length])
        latent1_prior_dists = map_distribution_structure(
            functools.partial(where_and_concat, latent1_reset_masks),
            latent1_first_prior_dists,
            latent1_after_first_prior_dists,
        )

        latent2_reset_masks = tf.tile(reset_masks[:, :, None], [1, 1, self.latent2_size])
        
        latent2_first_prior_dists = self.latent2_first_prior(latent1_posterior_samples)
        # these distributions start at t=1 and the last 2 inputs are from t-1
        latent2_after_first_prior_dists = self.latent2_prior(
            latent1_posterior_samples[:, 1:sequence_length + 1],
            latent2_posterior_samples[:, :sequence_length],
            actions[:, :sequence_length],
        )
        latent2_prior_dists = map_distribution_structure(
            functools.partial(where_and_concat, latent2_reset_masks),
            latent2_first_prior_dists,
            latent2_after_first_prior_dists,
        )

        outputs = {}

        if self.kl_analytic:
            latent1_kl_divergences = tfd.kl_divergence(latent1_posterior_dists, latent1_prior_dists)
        else:
            latent1_kl_divergences = (latent1_posterior_dists.log_prob(latent1_posterior_samples) - 
                                      latent1_prior_dists.log_prob(latent1_posterior_samples))
        latent1_kl_divergences = tf.reduce_sum(latent1_kl_divergences, axis=1)

        outputs.update({"latent1_kl_divergence": tf.reduce_mean(latent1_kl_divergences)})
        if self.latent2_posterior == self.latent2_prior: latent2_kl_divergences = 0.0
        else:
            if self.kl_analytic:
                latent2_kl_divergences = tfd.kl_divergence(latent2_posterior_dists, latent2_prior_dists)
            else:
                latent2_kl_divergences = (latent2_posterior_dists.log_prob(latent2_posterior_samples) -
                                          latent2_prior_dists.log_prob(latent2_posterior_samples))
            latent2_kl_divergences = tf.reduce_sum(latent2_kl_divergences, axis=1)

        outputs.update({"latent2_kl_divergence": tf.reduce_mean(latent2_kl_divergences),})
        outputs.update({"kl_divergence": tf.reduce_mean(latent1_kl_divergences + latent2_kl_divergences)})

        likelihood_dists = self.observation_decoder(latent1_posterior_samples, latent2_posterior_samples)
        likelihood_log_probs = likelihood_dists.log_prob(images)
        likelihood_log_probs = tf.reduce_sum(likelihood_log_probs, axis=1)

        image_mean_diffs = images - likelihood_dists.distribution.loc
        render_mean_diffs = image_mean_diffs #tf.minimum(tf.maximum(tf.abs(image_mean_diffs), 0), 255)
        reconstruction_error = tf.reduce_sum(
            tf.square(image_mean_diffs),
            axis=list(range(-len(likelihood_dists.event_shape), 0)),
        )
        reconstruction_error = tf.reduce_sum(reconstruction_error, axis=1)
        outputs.update({"log_likelihood": tf.reduce_mean(likelihood_log_probs),
                        "reconstruction_error": tf.reduce_mean(reconstruction_error), })

        # summed over the time dimension
        elbo = likelihood_log_probs - latent1_kl_divergences - latent2_kl_divergences

        if self.model_reward:
            # Tries to predict rewards!
            reward_dists = self.reward_predictor(
                latent1_posterior_samples[:, :sequence_length],
                latent2_posterior_samples[:, :sequence_length],
                actions[:, :sequence_length],
                latent1_posterior_samples[:, 1:sequence_length + 1],
                latent2_posterior_samples[:, 1:sequence_length + 1],
            )
            reward_valid_mask = tf.cast(
                tf.not_equal(step_types[:, :sequence_length], StepType.LAST),
                tf.float32,
            )
            reward_log_probs = reward_dists.log_prob(rewards[:, :sequence_length])
            reward_log_probs = tf.reduce_sum(
                reward_log_probs * reward_valid_mask, axis=1
            )
            reward_reconstruction_error = tf.square(
                rewards[:, :sequence_length] - reward_dists.loc
            )
            reward_reconstruction_error = tf.reduce_sum(
                reward_reconstruction_error * reward_valid_mask, axis=1
            )
            outputs.update(
                {
                    "reward_log_likelihood": tf.reduce_mean(reward_log_probs),
                    "reward_reconstruction_error": tf.reduce_mean(
                        reward_reconstruction_error
                    ),
                }
            )
            elbo += reward_log_probs

        if self.model_discount:
            discount_dists = self.discount_predictor(
                latent1_posterior_samples[:, 1 : sequence_length + 1],
                latent2_posterior_samples[:, 1 : sequence_length + 1],
            )
            discount_log_probs = discount_dists.log_prob(discounts[:, :sequence_length])
            discount_log_probs = tf.reduce_sum(discount_log_probs, axis=1)
            discount_accuracy = tf.cast(
                tf.equal(
                    tf.cast(discount_dists.mode(), tf.float32),
                    discounts[:, :sequence_length],
                ),
                tf.float32,
            )
            discount_accuracy = tf.reduce_sum(discount_accuracy, axis=1)
            outputs.update(
                {
                    "discount_log_likelihood": tf.reduce_mean(discount_log_probs),
                    "discount_accuracy": tf.reduce_mean(discount_accuracy),
                }
            )
            elbo += discount_log_probs

        # average over the batch dimension
        loss = -tf.reduce_mean(elbo)

        posterior_images = likelihood_dists.mean()
        prior_images = self.observation_decoder(latent1_prior_samples, latent2_prior_samples).mean()
        conditional_prior_images = self.observation_decoder(
            latent1_conditional_prior_samples, latent2_conditional_prior_samples
        ).mean()

        outputs.update(
            {
                "elbo": tf.reduce_mean(elbo),
                "images": images,
                "posterior_images": posterior_images,
                "prior_images": prior_images,
                "image_mean_diffs": render_mean_diffs,
                "conditional_prior_images": conditional_prior_images,
            }
        )
        return loss, outputs

    def sample_prior_or_posterior(self, actions, step_types=None, images=None):
        """Samples from the prior, except for the time steps in which conditioning images are given.

        Samples the posteriors p(z_{t+1} | a_{1:t}, x_{1:t})) OR
        Samples the priors     p(z_{t+1} | a_{1:t}))

        It returns lists of intermediate distributions used for sampling, as well as those samples:

        z_{t+1} ~ p(z_{t+1} | a_t, z_t, [x_t])

        Because the model has two layers, the samples and distributions are:

        z^1_{t+1} ~ p(z_{t+1}^1 | z_t^2, a_t, [x_{t+1}])         [Latent1 prior or posterior]
        z^2_{t+1} ~ p(z_{t+1}^2 | a_t, z_t^2, z_{t+1}^1)         [Latent2 prior or posterior]

        :param actions: (B, T-1, A)
        :param step_types: (B, T)
        :param images: (B, ?, ..., ?) (optional) If provided, the first distribution will be the posterior instead of the prior.

        """
        # semihack to handle discrete actions
        actions = tf.cast(actions, dtype=tf.float32)

        if step_types is None:
            # Create step types if they're not provided, assume they're all MID-type. [TODO seems risky assumption]
            batch_size = tf.shape(actions)[0]
            sequence_length = actions.shape[1].value  # should be statically defined
            step_types = tf.fill([batch_size, sequence_length + 1], StepType.MID)
        else:
            # Clip the actions to the correct length (if necessary).
            # sequence_length = T-1
            sequence_length = step_types.shape[1].value - 1
            actions = actions[:, :sequence_length]

        # Compute features from the images, if provided.
        if images is not None: features = self.compressor(images)

        # Swap batch and time axes.
        actions = tf.transpose(actions, [1, 0, 2])
        step_types = tf.transpose(step_types, [1, 0])
        if images is not None:
            features = tf.transpose(features, [1, 0, 2])

        # Collect the sequences of distributions and samples.
        latent1_dists = []
        latent1_samples = []
        latent2_dists = []
        latent2_samples = []

        # This means: for t in range(T)
        for t in range(sequence_length + 1):
            # Make the next distribution conditional if we have images and? [TODO comment]
            is_conditional = images is not None and (t < images.shape[1].value)
            if t == 0:
                # If conditional, create the first-level posterior.
                # q(z_1^1 | x_1). Receives the image!
                if is_conditional: latent1_dist = self.latent1_first_posterior(features[t])
                # If unconditional, create the first-level prior.
                # p(z_1^1)
                else: latent1_dist = self.latent1_first_prior(step_types[t])  

                # Sample from the first-level distribution.
                latent1_sample = latent1_dist.sample()

                # Create the second-level distribution.
                # The posterior depends on the image indirectly through the latent1 first posterior.
                # q(z_1^2 | z_1^1) = p(z_1^2 | z_1^1)
                if is_conditional: latent2_dist = self.latent2_first_posterior(latent1_sample)
                # p(z_1^2 | z_1^1)
                else: latent2_dist = self.latent2_first_prior(latent1_sample)

                # Sample from the second-level distribution.
                latent2_sample = latent2_dist.sample()
            else:
                reset_mask = tf.equal(step_types[t], StepType.FIRST)
                if is_conditional:
                    # q(z_1^1 | x_1). Receives the image! Althought t > 0, we may need this depending on the reset mask.
                    latent1_first_dist = self.latent1_first_posterior(features[t])
                    # q(z_{t+1}^1 | x_{t+1}, z_t^2, a_t)
                    latent1_dist = self.latent1_posterior(features[t], latent2_samples[t - 1], actions[t - 1])
                else:
                    # p(z_1^1). Althought t > 0, we may need this depending on the reset mask.
                    latent1_first_dist = self.latent1_first_prior(step_types[t])
                    latent1_dist = self.latent1_prior(latent2_samples[t - 1], actions[t - 1])
                latent1_dist = map_distribution_structure(
                    functools.partial(tf.where, reset_mask),
                    latent1_first_dist,
                    latent1_dist,
                )
                latent1_sample = latent1_dist.sample()

                if is_conditional:
                    latent2_first_dist = self.latent2_first_posterior(latent1_sample)
                    latent2_dist = self.latent2_posterior(
                        latent1_sample, latent2_samples[t - 1], actions[t - 1]
                    )
                else:
                    latent2_first_dist = self.latent2_first_prior(latent1_sample)
                    latent2_dist = self.latent2_prior(
                        latent1_sample, latent2_samples[t - 1], actions[t - 1]
                    )
                latent2_dist = map_distribution_structure(
                    functools.partial(tf.where, reset_mask),
                    latent2_first_dist,
                    latent2_dist,
                )
                latent2_sample = latent2_dist.sample()

            latent1_dists.append(latent1_dist)
            latent1_samples.append(latent1_sample)
            latent2_dists.append(latent2_dist)
            latent2_samples.append(latent2_sample)

        latent1_dists = map_distribution_structure(
            lambda *x: tf.stack(x, axis=1), *latent1_dists
        )
        # (B T, latent1_dim)
        latent1_samples = tf.stack(latent1_samples, axis=1)
        latent2_dists = map_distribution_structure(
            lambda *x: tf.stack(x, axis=1), *latent2_dists
        )
        # (B, T, latent2_dim)
        latent2_samples = tf.stack(latent2_samples, axis=1)
        return (latent1_samples, latent2_samples), (latent1_dists, latent2_dists)

    def sample_posterior(self, images, actions, step_types, features=None):
        """
        q(z_{t+1} | x_{t+1}, z_t, a_t) for each image

        :param images: 
        :param actions: 
        :param step_types: 
        :param features: 
        :returns: 
        :rtype: 

        """
        # semihack for discrete actions
        actions = tf.cast(actions, dtype=tf.float32)
        
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
                # This handles the case of the stype types changing from MID/END to FIRST
                reset_mask = tf.equal(step_types[t], StepType.FIRST)
                latent1_first_dist = self.latent1_first_posterior(features[t])

                # Create the latent1-posterior based on the features (images), the most-recently sampled z from layer 2, and the most recent action
                latent1_dist = self.latent1_posterior(
                    features[t], latent2_samples[t - 1], actions[t - 1]
                )
                latent1_dist = map_distribution_structure(
                    functools.partial(tf.where, reset_mask),
                    latent1_first_dist,
                    latent1_dist,
                )
                latent1_sample = latent1_dist.sample()

                latent2_first_dist = self.latent2_first_posterior(latent1_sample)

                # Create the latent2-posterior based on the features (images), the most-recently sampled z from both layers,
                #    and the most recent action
                latent2_dist = self.latent2_posterior(
                    latent1_sample, latent2_samples[t - 1], actions[t - 1]
                )
                # TODO fix so it doesn't destroy our names.
                latent2_dist = map_distribution_structure(
                    functools.partial(tf.where, reset_mask),
                    latent2_first_dist,
                    latent2_dist,
                )
                latent2_sample = latent2_dist.sample()

            latent1_dists.append(latent1_dist)
            latent1_samples.append(latent1_sample)
            latent2_dists.append(latent2_dist)
            latent2_samples.append(latent2_sample)

        latent1_dists = map_distribution_structure(
            lambda *x: tf.stack(x, axis=1), *latent1_dists
        )
        latent1_samples = tf.stack(latent1_samples, axis=1)
        latent2_dists = map_distribution_structure(
            lambda *x: tf.stack(x, axis=1), *latent2_dists
        )
        latent2_samples = tf.stack(latent2_samples, axis=1)
        return (latent1_samples, latent2_samples), (latent1_dists, latent2_dists)

    def compute_future_observation_likelihoods(self, actions, step_types, images):
        """ Estimate:
              p(x_T=future_image | a_{1:T-1}=actions, x_{1:T-1}=past_images) 
                =E_{z_{T} ~ p(z_{T} | x_{1:T-1}, a_{1:T-1})} p(x_{T} | z_{T} )

        :param actions: (B, T-1, A) action sequence
        :param step_types: (B, T) step-type sequence
        :param images: (B, T, ...) observed images sequence
        :returns: 
        :rtype: 
        """

        x_1toTm1 = images[:, :-1]
        x_T = images[:, -1]
        del images

        # Sample z_{t+1} ~ p(z_{t+1} | x_{1:t}, a_{1:t})
        # TODO compute a bunch of z_{t+1}! Not just one. This will better-estimate the expectation. 
        ((l1_samples, l2_samples), _) = (self.sample_prior_or_posterior(actions=actions, step_types=step_types, images=x_1toTm1))

        # (z^1_T, z^2_T)
        last_latents = (l1_samples[:, -1], l2_samples[:, -1])

        # Approximation of p(x_T | x_{1:T-1}, a_{1:T}). Not exact since the latents are samples.
        # p(x_{T}=future_image | a_{1:T-1}=actions, x_{1:T-1}=past_images)  =
        #    E_{z_{T} ~ p(z_{T} | x_{1:T-1}, a_{1:T-1})} p(x_{T} | z_{T})
        approx_p_xT_pdf = self.observation_decoder(*last_latents)

        # (B,) Likelihoods of the last image, x_T.
        approx_log_p_xT_value = approx_p_xT_pdf.log_prob(x_T)
        return approx_log_p_xT_value

    # def compute_future_latent_log_prob(self, actions, step_types, images):
        # latent1_dist.log_prob(
    
    def _parser(self, images, actions):
        num_shifts = self._all_sequence_length - self._sequence_length
        t_start = tf.random_uniform([], 0, num_shifts + 1, dtype=tf.int32)
        images = images[t_start:t_start+self._sequence_length]
        images.set_shape([self._sequence_length] + images.shape.as_list()[1:])
        actions = actions[t_start:t_start+self._sequence_length-1]
        actions.set_shape([self._sequence_length-1] + actions.shape.as_list()[1:])
        seqs = {
            'images': images,
            'actions': actions,
        }
        return seqs  
    
    def parser(self, images, actions):
        ### I think this code randomly parses out a sequence from the full sequence
        num_shifts = self._all_sequence_length - self._sequence_length
        t_start = int(np.random.uniform(0, num_shifts + 1, size=1)[0])
        images = images[t_start:t_start+self._sequence_length]
        images.set_shape([self._sequence_length] + images.shape.as_list()[1:])
        actions = actions[t_start:t_start+self._sequence_length-1]
        actions.set_shape([self._sequence_length-1] + actions.shape.as_list()[1:])
        seqs = {
            'images': images,
            'actions': actions,
        }
        return seqs  
        

    def compile(self):
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self._sess = tf.Session(config=config)
        img_size = self.getSettings()["fd_terrain_shape"]
        action_size = 3
        reward_size = 1
        data_array = np.zeros((64,16,np.prod(img_size) + action_size + reward_size))
        # data_array = np.concatenate([data1, data2, data4, data5,data6,data7])
        
        data_array = data_array.astype(np.float32)
        self._all_batch_size, self._all_sequence_length = data_array.shape[:2]
        self._batch_size = 32
        self._sequence_length = 8
        self._states_placeholder = tf.placeholder(shape=[32, self._sequence_length] + self.getSettings()["fd_terrain_shape"], dtype=tf.float32)
        self._action_placeholder = tf.placeholder(shape=[32, self._sequence_length, action_size], dtype=tf.float32)
        shuffle = True
        num_epochs = None
        dataset = tf.data.Dataset.from_tensor_slices((data_array[..., :np.prod(img_size)].reshape(data_array.shape[:2] + tuple(img_size)), data_array[..., np.prod(img_size):np.prod(img_size)+action_size]))
        
        if shuffle:
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1024, count=num_epochs))
        else:
            dataset = dataset.repeat(num_epochs)
        
        dataset = dataset.apply(tf.data.experimental.map_and_batch(self._parser, self._batch_size, drop_remainder=True))
        dataset = dataset.prefetch(self._batch_size)
        
        iterator = dataset.make_one_shot_iterator()
        data = iterator.get_next()
# #         step_types = tf.fill(tf.shape(data['images'])[:2], StepType.MID)
#         data = {}
#         ### 100 trajectories of length 100 for 64x64x3 images
#         data["images"] = np.zeros((10,8,64,64,3))
#         ### 100 trajectories of length 100 for 3 actions
#         data["actions"] = np.zeros((10,8,3))
        step_types = tf.fill(tf.shape(data['images'])[:2], StepType.MID)
        self._loss, self._outputs = self.compute_loss(self._states_placeholder, self._action_placeholder, step_types)
        
        self._global_step = tf.train.create_global_step()
        self._adam_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        self._train_op = self._adam_optimizer.minimize(self._loss, global_step=self._global_step)
        
        # train
        self._sess.run(tf.global_variables_initializer()) 
#         if not args.init_params is None:
#           load_trainable_weights(sess, args.init_params)
        
        
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
        pass
#         self._model.reset()
#         self._model._reward_net.reset_states()
#         self._model._forward_dynamics_net.reset_states()
#         if not (self._modelTarget is None):
#             self._modelTarget._forward_dynamics_net.reset_states()
            # self._modelTarget._reward_net.reset_states()
            # self._modelTarget.reset()
            
    def getNetworkParameters(self):
        params = []
#         params.append(copy.deepcopy(self._model._forward_dynamics_net.get_weights()))
#         params.append(copy.deepcopy(self._model._reward_net.get_weights()))
#         
#         if ( "return_rnn_sequence" in self.getSettings()
#              and (self.getSettings()["return_rnn_sequence"])):
#             params.append(copy.deepcopy(self._model._reward_net_seq.get_weights()))
                
        return params
    
    def setNetworkParameters(self, params):
#         self._model._forward_dynamics_net.set_weights(params[0])
#         self._model._reward_net.set_weights(params[1])
#         if ( "return_rnn_sequence" in self.getSettings()
#              and (self.getSettings()["return_rnn_sequence"])):
#             self._model._reward_net_seq.set_weights(params[2])
        pass
        
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
                
    def train(self, states, actions, result_states, rewards, falls=None, updates=1, batch_size=None, p=1, lstm=True, datas=None, trainInfo=None):
        """
            states will come for the agent and
            results_states can come from the imitation agent
        """
        # print ("fd: ", self)
        # print ("state length: ", len(self.getStateBounds()[0]))
        update_data = True
        if (update_data):
            img_size = self.getSettings()["fd_terrain_shape"]
            action_size = 3
            reward_size = 1
            # data_array = np.zeros((32,16,np.prod(img_size) + action_size + reward_size))
            # data_array = np.concatenate([data1, data2, data4, data5,data6,data7])
            data_array = states
            
            self._batch_size = 32
            self._sequence_length = 8
            self.reset()
            shuffle = True
            num_epochs = None
            states = states.reshape(data_array.shape[:2] + tuple(img_size))
        
        if ('anneal_learning_rate' in self.getSettings()
            and (self.getSettings()['anneal_learning_rate'] == True)):
            K.set_value(self._model._forward_dynamics_net.optimizer.lr, np.float32(self.getSettings()['fd_learning_rate']) * p)

        for i in range(1):
            out, loss_val, global_step_val = self._sess.run([self._train_op,self._loss, self._global_step], 
                                                        feed_dict={self._states_placeholder: states, self._action_placeholder: actions})
          
#         print('step = %d, loss = %f' % (global_step_val, loss_val))
#         print ("trainInfo: ", trainInfo)
#           if i % 100 == 0:
#         if trainInfo["round"] % 5 == 0 and (trainInfo["epoch"] == 0) and (trainInfo["iteration"] == 0) :
#           images, posterior_images, conditional_prior_images, prior_images = self._sess.run(
#               [self._outputs['images'], self._outputs['posterior_images'], self._outputs['conditional_prior_images'], self._outputs['prior_images']],
#                                                         feed_dict={self._states_placeholder: states, self._action_placeholder: actions})
#           all_images = np.concatenate([images, posterior_images, conditional_prior_images, prior_images], axis=2)
#           save_weights(self._sess, "data/", counter=trainInfo["round"])
#           #import ipdb;ipdb.set_trace()
#           display_gif(all_images, "data/", counter=trainInfo["round"])
        return loss_val
    
    def predict_encoding(self, state):
        """
            Compute distance between two states
        """
        # state = np.array(norm_state(state, self.getStateBounds()), dtype=self.getSettings()['float_type'])
        if (("train_LSTM_FD" in self._settings)
                    and (self._settings["train_LSTM_FD"] == True)):
            h_a = self._model.processed_a.predict([np.array([state])])#         self.reset()
#         hf = h5py.File(fileName+"_bounds.h5", "w")
#         hf.create_dataset('_state_bounds', data=self.getStateBounds())
#         hf.create_dataset('_reward_bounds', data=self.getRewardBounds())
#         hf.create_dataset('_action_bounds', data=self.getActionBounds())
#         if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
#             print("fd save self.getStateBounds(): ", len(self.getStateBounds()[0]))
#         # hf.create_dataset('_resultgetStateBounds()', data=self.getResultStateBounds())
#         # print ("fd: ", self)
#         hf.flush()
#         hf.close()
#         suffix = ".h5"
#         ### Save models
#         # self._model._actor_train.save(fileName+"_actor_train"+suffix, overwrite=True)
#         self._model._forward_dynamics_net.save(fileName+"_FD"+suffix, overwrite=True)
#         # self._model._reward_net.save(fileName+"_reward"+suffix, overwrite=True)
#         self._model._reward_net.save_weights(fileName+"_reward"+suffix, overwrite=True)
#         self._modelTarget._forward_dynamics_net.save(fileName+"_FD_T"+suffix, overwrite=True)
#         # self._modelTarget._forward_dynamics_net.save(fileName+"_FD_T"+suffix, overwrite=True)
#         # self._model._reward_net.save(fileName+"_reward"+suffix, overwrite=True)
#         # self._modelTarget._reward_net.save_weights(fileName+"_reward_T"+suffix, overwrite=True)
#         # print ("self._model._actor_train: ", self._model._actor_train)
#         try:
#             from keras.utils import plot_model
#             ### Save model design as image
#             plot_model(self._model._forward_dynamics_net, to_file=fileName+"_FD"+'.svg', show_shapes=True)
#             plot_model(self._model._reward_net, to_file=fileName+"_reward"+'.svg', show_shapes=True)
#             plot_model(self._modelTarget._forward_dynamics_net, to_file=fileName+"_FD_decode"+'.svg', show_shapes=True)
#         except Exception as inst:
#             ### Maybe the needed libraries are not available
#             print ("Error saving diagrams for rl models.")
#             print (inst)
#             
#         if (states is not None):
#             ## Don't use Xwindows backend for this
#             import matplotlib
#             # matplotlib.use('Agg')
#             import matplotlib.pyplot as plt
#             # img_ = np.reshape(viewData, (150,158,3))
#             ### get the sequence prediction
#             predicted_y = self._model._reward_net.predict([states, actions], batch_size=states.shape[0])
#             
#             img_ = predicted_y[0]
#             img_z = predicted_y[1]
#             ### Get first sequence in batch
#             img_ = img_[0]
#             img_x = states[0]
#             import imageio
#             import PIL
#             from PIL import Image
#             images_x = []
#             images_y = []
#             images_z = []
#             for i in range(len(img_)):
#                 img__y = np.reshape(img_[i], self._settings["fd_terrain_shape"])
#                 images_y.append(Image.fromarray(img__y).resize((256,256))) ### upsampling
#                 print("img_ shape", img__y.shape, " sum: ", np.sum(img__y))
#                 # fig1 = plt.figure(2)
#                 ### Save generated image
#                 # plt.imshow(img__y, origin='lower')
#                 # plt.title("agent visual Data: ")
#                 # fig1.savefig(fileName+"viz_state_"+str(i)+".png")
#                 ### Save input image
#                 img__x = np.reshape(img_x[i], self._settings["fd_terrain_shape"])
#                 images_x.append(Image.fromarray(img__x).resize((256,256))) ### upsampling
# #                 plt.imshow(img__x, origin='lower')
# #                 plt.title("agent visual Data: ")
# #                 fig1.savefig(fileName+"viz_state_input_"+str(i)+".png")
# #                 img__z = np.reshape(img_z[i], self._settings["fd_terrain_shape"])
# #                 images_z.append(img__z)
#                 
#             imageio.mimsave(fileName+"viz_state_input_"+'.gif', images_x, duration=0.5,)
#             imageio.mimsave(fileName+"viz_conditional_"+'.gif', images_y, duration=0.5,)
# #             imageio.mimsave(fileName+"viz_marginal_"+'.gif', images_z, duration=0.5,)
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
            errors=[0]
#             predicted_y = self._model._reward_net.predict([states, actions], batch_size=states.shape[0])
#             errors.append( np.mean(predicted_y[0] - states ))
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
        states = states.reshape(states.shape[:2] + tuple(self.getSettings()["fd_terrain_shape"]))
        images, posterior_images, conditional_prior_images, prior_images = self._sess.run(
            [self._outputs['images'], self._outputs['posterior_images'], self._outputs['conditional_prior_images'], self._outputs['prior_images']],
                                                      feed_dict={self._states_placeholder: states, self._action_placeholder: actions})
        all_images = np.concatenate([images, posterior_images, conditional_prior_images, prior_images], axis=2)
        self.save_weights(self._sess, fileName+"_slac_model", counter=0)
        #import ipdb;ipdb.set_trace()
        display_gif(all_images, fileName+"_slac_model", counter=0)
#         self.reset()
#         hf = h5py.File(fileName+"_bounds.h5", "w")
#         hf.create_dataset('_state_bounds', data=self.getStateBounds())
#         hf.create_dataset('_reward_bounds', data=self.getRewardBounds())
#         hf.create_dataset('_action_bounds', data=self.getActionBounds())
#         if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
#             print("fd save self.getStateBounds(): ", len(self.getStateBounds()[0]))
#         # hf.create_dataset('_resultgetStateBounds()', data=self.getResultStateBounds())
#         # print ("fd: ", self)
#         hf.flush()
#         hf.close()
#         suffix = ".h5"
#         ### Save models
#         # self._model._actor_train.save(fileName+"_actor_train"+suffix, overwrite=True)
#         self._model._forward_dynamics_net.save(fileName+"_FD"+suffix, overwrite=True)
#         # self._model._reward_net.save(fileName+"_reward"+suffix, overwrite=True)
#         self._model._reward_net.save_weights(fileName+"_reward"+suffix, overwrite=True)
#         self._modelTarget._forward_dynamics_net.save(fileName+"_FD_T"+suffix, overwrite=True)
#         # self._modelTarget._forward_dynamics_net.save(fileName+"_FD_T"+suffix, overwrite=True)
#         # self._model._reward_net.save(fileName+"_reward"+suffix, overwrite=True)
#         # self._modelTarget._reward_net.save_weights(fileName+"_reward_T"+suffix, overwrite=True)
#         # print ("self._model._actor_train: ", self._model._actor_train)
#         try:
#             from keras.utils import plot_model
#             ### Save model design as image
#             plot_model(self._model._forward_dynamics_net, to_file=fileName+"_FD"+'.svg', show_shapes=True)
#             plot_model(self._model._reward_net, to_file=fileName+"_reward"+'.svg', show_shapes=True)
#             plot_model(self._modelTarget._forward_dynamics_net, to_file=fileName+"_FD_decode"+'.svg', show_shapes=True)
#         except Exception as inst:
#             ### Maybe the needed libraries are not available
#             print ("Error saving diagrams for rl models.")
#             print (inst)
#             
#         if (states is not None):
#             ## Don't use Xwindows backend for this
#             import matplotlib
#             # matplotlib.use('Agg')
#             import matplotlib.pyplot as plt
#             # img_ = np.reshape(viewData, (150,158,3))
#             ### get the sequence prediction
#             predicted_y = self._model._reward_net.predict([states, actions], batch_size=states.shape[0])
#             
#             img_ = predicted_y[0]
#             img_z = predicted_y[1]
#             ### Get first sequence in batch
#             img_ = img_[0]
#             img_x = states[0]
#             import imageio
#             import PIL
#             from PIL import Image
#             images_x = []
#             images_y = []
#             images_z = []
#             for i in range(len(img_)):
#                 img__y = np.reshape(img_[i], self._settings["fd_terrain_shape"])
#                 images_y.append(Image.fromarray(img__y).resize((256,256))) ### upsampling
#                 print("img_ shape", img__y.shape, " sum: ", np.sum(img__y))
#                 # fig1 = plt.figure(2)
#                 ### Save generated image
#                 # plt.imshow(img__y, origin='lower')
#                 # plt.title("agent visual Data: ")
#                 # fig1.savefig(fileName+"viz_state_"+str(i)+".png")
#                 ### Save input image
#                 img__x = np.reshape(img_x[i], self._settings["fd_terrain_shape"])
#                 images_x.append(Image.fromarray(img__x).resize((256,256))) ### upsampling
# #                 plt.imshow(img__x, origin='lower')
# #                 plt.title("agent visual Data: ")
# #                 fig1.savefig(fileName+"viz_state_input_"+str(i)+".png")
# #                 img__z = np.reshape(img_z[i], self._settings["fd_terrain_shape"])
# #                 images_z.append(img__z)
#                 
#             imageio.mimsave(fileName+"viz_state_input_"+'.gif', images_x, duration=0.5,)
#             imageio.mimsave(fileName+"viz_conditional_"+'.gif', images_y, duration=0.5,)
# #             imageio.mimsave(fileName+"viz_marginal_"+'.gif', images_z, duration=0.5,)

    def save_weights(self, sess, logdir, counter):
        import pickle
        vars_dict = {}
        graph_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#         for var in graph_vars:
#             vars_dict[var.name]= self._sess.run(var)
        fobj = open(logdir+str(counter)+'-weights.pkl', 'wb')
        pickle.dump(vars_dict , fobj)
    
    def load_trainable_weights(self, sess, pathname):
        import pickle
        load_data = pickle.load(open(pathname, 'rb')) #sorry
        assign_ops = [tf.assign(var, load_data[var.name]) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
        for op in assign_ops:
            self._sess.run(op)
        
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
        