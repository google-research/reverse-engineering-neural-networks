# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Define simple learned optimizer models."""

import numbers

import jax
from jax.experimental import optimizers
from jax.experimental import stax
import jax.numpy as jnp

# Aliases for standard initializers and nonlinearities.
fan_in = jax.nn.initializers.variance_scaling(1., 'fan_in', 'normal')
zeros = jax.nn.initializers.zeros


def append_to_sequence(sequence, element):
  """Appends an element to a rolling sequence buffer.

  Args:
    sequence: a sequence of ndarrays, concatenated along the first dimension.
    element: an ndarray to add to the sequence.

  Returns:
    sequence: the updated sequence, with the first element removed, the rest
      of the elements shifted over, and the new element added.
  """
  return jnp.vstack((sequence[1:], element[jnp.newaxis, ...]))


def cwrnn(key, cell, input_scale='raw', output_scale=1e-3):
  """Component-wise RNN Optimizer.

  This optimizer applies an RNN to update the parameters of each problem
  variable independently (hence the name, component-wise). It follows the
  same approach as in previous work (Andrychowicz et al 2016, Wichrowska
  et al 2017) that distribute the parameters along the batch dimension
  of the RNN. This allows us to easily update each parameter in parallel.

  Args:
    key: Jax PRNG key to use for initializing parameters.
    cell: An RNNCell to use (see renn/rnn/cells.py)
    input_scale: str, Specifies how to scale gradient inputs to the RNN. If
      'raw', then the gradients are not scaled. If 'log1p', then the scale
      and the sign of the inputs are split into a length 2 vector,
      [log1p(abs(g)), sign(g)].
    output_scale: float, Constant used to multiply (rescale) the RNN output.

  Returns:
    meta_parameters: A tuple containing the RNN parameters and the readout
      parameters. The RNN parameters themselves are a namedtuple. The readout
      parameters are also a tuple containing weights and a bias.

    optimizer_fun: A function that takes a set of meta_parameters and
      initializes an optimizer tuple containing functions to initialize the
      optimizer state, update the optimizer state, and get parameters from
      the optimizer state.
  """
  # Input and output shapes.
  n_in = 2 if input_scale == 'log1p' else 1
  n_out = 1

  # Initialize the readout
  readout_init, readout_apply = stax.Dense(n_out, W_init=zeros, b_init=zeros)

  # Initialize parameters.
  rnn_key, readout_key = jax.random.split(key)
  rnn_shape, rnn_params = cell.init(rnn_key, (None, n_in))
  _, readout_params = readout_init(readout_key, rnn_shape)
  initial_theta = (rnn_params, readout_params)

  @jax.experimental.optimizers.optimizer
  def optimizer_fun(theta):
    """Builds a component-wise RNN optimizer."""
    rnn_params, readout_params = theta

    def init_state(x):
      n = jnp.ravel(x).size
      return (x, cell.get_initial_state(rnn_params, batch_size=n))

    def update_opt(_, grads, state):
      x, h = state

      grad_vec = jnp.reshape(grads, (-1, 1))

      # Inputs are scaled by a constant factor.
      if isinstance(input_scale, numbers.Number):
        inputs = input_scale * grad_vec

      # Inputs are raw (unmodified) gradients.
      elif input_scale == 'raw':
        inputs = grad_vec

      # Inputs are the log-scale and sign of the gradient.
      elif input_scale == 'log1p':
        scale = jnp.log1p(jnp.abs(grad_vec))
        sign = jnp.sign(grad_vec)
        inputs = jnp.hstack((scale, sign))

      else:
        raise ValueError(f'Invalid input scale {input_scale}.')

      h_next = cell.batch_apply(rnn_params, inputs, h)
      outputs = readout_apply(readout_params, h_next)
      x_next = x + output_scale * jnp.reshape(outputs, x.shape)
      return (x_next, h_next)

    def get_params(state):
      return state[0]

    return (init_state, update_opt, get_params)

  return initial_theta, optimizer_fun


def lds(key, num_units, h0_init=zeros, w_init=fan_in):
  """Linear dynamical system (LDS) optimizer."""
  hstar_key, rec_key, inp_key, readout_key = jax.random.split(key, 4)

  # Initialize linear dynamical system
  h0 = h0_init(hstar_key, (num_units,))
  rec_jac = w_init(rec_key, (num_units, num_units))
  inp_jac = w_init(inp_key, (num_units, num_units))

  # Initialize the readout
  readout_init, readout_apply = stax.Dense(1, W_init=zeros, b_init=zeros)
  _, readout_params = readout_init(readout_key, (None, num_units))

  initial_meta_params = (h0, rec_jac, inp_jac, readout_params)

  @optimizers.optimizer
  def optimizer_fun(h_star, g_star, h_init, rec_jac, inp_jac, readout_params):
    """Linear dynamical system optimizer.

    Args:
      h_star: The state around which to linearize.
      g_star: The input around which to linearize.
      h_init: The initial state.
      rec_jac: Defines the recurrent dynamics.
      inp_jac: Multiplies the input gradients.
      readout_params: Tuple of (weights, biases).

    Returns:
      init_state: Initialize the optimizer state.
      update_opt: Updates the optimizer state variables given the current
        step, gradients, and current state.
      get_params: Gets parameters from the optimizer state.
    """

    def init_state(x):
      batch_size = jnp.ravel(x).size
      h = jnp.ones((batch_size, 1)) * jnp.reshape(h_init, (1, -1))
      return (x, h)

    def update_opt(_, grads, state):
      x, h = state
      g = jnp.reshape(grads, (-1, 1))
      h_next = h_star + jnp.dot(h - h_star, rec_jac.T) + jnp.dot(g - g_star, inp_jac.T)  # pylint: disable=line-too-long
      outputs = readout_apply(readout_params, h_next)
      x_next = x + jnp.reshape(outputs, x.shape)
      return (x_next, h_next)

    def get_params(state):
      return state[0]

    return (init_state, update_opt, get_params)

  return initial_meta_params, optimizer_fun


def linear(key, tau, scale, base=0):
  """Optimizer that is a linear function of gradient history."""
  initial_meta_params = base + scale * jax.random.uniform(key, (tau,))

  @optimizers.optimizer
  def optimizer_fun(meta_params):
    """Builds a linear optimizer with the given meta_params."""

    def init_fun(params):
      """Initialize optimizer state."""
      grad_seq = jnp.zeros((tau,) + params.shape)
      return (params, grad_seq)

    def update_fun(step, grads, state):
      """Apply a step of the optimizer."""
      del step  # Unused.
      params, grad_seq = state
      grad_seq = append_to_sequence(grad_seq, grads)
      params -= jnp.tensordot(meta_params, grad_seq, axes=1)
      return (params, grad_seq)

    def get_params(state):
      """Get parameters from the optimizer."""
      return state[0]

    return (init_fun, update_fun, get_params)

  return initial_meta_params, optimizer_fun


def linear_dx(key, tau, scale_grad, scale_dx, base_grad=0, base_gram=0):
  """Optimizer that is a linear function of gradient and parameter history."""

  key0, key1 = jax.random.split(key, 2)
  initial_meta_params = (base_grad +
                         scale_grad * jax.random.uniform(key0,
                                                         (tau,)), base_gram +
                         scale_dx * jax.random.uniform(key1, (tau - 1,)))

  @optimizers.optimizer
  def optimizer_fun(meta_params):
    """Builds a linear_dx optimizer."""
    theta_grad, theta_dx = meta_params

    def init_fun(params):
      """Initialize optimizer state."""
      grad_seq = jnp.zeros((tau,) + params.shape)
      param_seq = jnp.zeros((tau,) + params.shape)
      return (params, grad_seq, param_seq)

    def update_fun(step, grads, state):
      """Apply a step of the optimizer."""
      del step  # Unused.
      params, grad_seq, param_seq = state
      grad_seq = append_to_sequence(grad_seq, grads)
      param_seq = append_to_sequence(param_seq, params)

      # Differences in parameters.
      # TODO(nirum): This recomputes differences at every iteration. Should
      # time this to ensure that the repeated jnp.diff call is not too slow.
      delta_params = jnp.diff(param_seq, axis=0)

      grad_term = jnp.tensordot(theta_grad, grad_seq, axes=1)
      dx_term = jnp.tensordot(theta_dx, delta_params, axes=1)
      params -= (grad_term + dx_term)

      return (params, grad_seq, param_seq)

    def get_params(state):
      return state[0]

    return init_fun, update_fun, get_params

  return initial_meta_params, optimizer_fun


def gradgram(key, tau, scale_grad, scale_gram, base_grad=0, base_gram=0):
  """Optimizer that is a function of gradient history and inner products."""

  # Initialize meta-parameters.
  key0, key1 = jax.random.split(key, 2)

  initial_meta_params = (base_grad +
                         scale_grad * jax.random.uniform(key0,
                                                         (tau,)), base_gram +
                         scale_gram * jax.random.uniform(key1, (tau,)))

  # Generalized inner product.
  innerprod = jax.jit(
      jax.vmap(jax.vmap(lambda x, y: -jnp.sum(x * y), in_axes=(0, None)),
               in_axes=(None, 0)))

  # Batched norm.
  norms = jax.jit(jax.vmap(jnp.linalg.norm, in_axes=0))

  @optimizers.optimizer
  def optimizer_fun(meta_params):
    """An optimizer that uses gradient-gradient correlations."""
    theta_grad, theta_gram = meta_params

    def init_fun(params):
      """Initialize the optimizer state."""
      grad_seq = jnp.zeros((tau,) + params.shape)
      return (params, grad_seq)

    def update_fun(step, grads, state):
      """Apply a step of the optimzier."""
      del step  # Unused.
      params, grad_seq = state

      # Update gradient history.
      grad_seq = append_to_sequence(grad_seq, grads)

      # Compute normalized gram matrix.
      gram = innerprod(grad_seq, grad_seq)
      grad_norm = norms(grad_seq)
      gram /= (jnp.outer(grad_norm, grad_norm) + 1e-6)

      # Compute update terms.
      attn_weights = jnp.dot(stax.softmax(gram, axis=0), theta_gram)
      attn_term = jnp.tensordot(attn_weights, grad_seq, axes=1)
      grad_term = jnp.tensordot(theta_grad, grad_seq, axes=1)
      params -= (grad_term + attn_term)

      return (params, grad_seq)

    def get_params(state):
      return state[0]

    return init_fun, update_fun, get_params

  return initial_meta_params, optimizer_fun


def momentum(key):
  """Wrapper for the momentum optimizer."""
  del key  # Unused.
  initial_learning_rate = 1e-3
  initial_mass = 0.8

  def optimizer_fun(optimizer_params):
    return optimizers.momentum(*optimizer_params)

  return (initial_learning_rate, initial_mass), optimizer_fun


def aggmo(key, num_terms):
  """Aggregated momentum (aggmo)."""
  initial_learning_rate = 0.0
  initial_masses = zeros(key, (num_terms,))
  initial_meta_params = (initial_learning_rate, initial_masses)

  @optimizers.optimizer
  def optimizer_fun(v0, alphas, betas):
    """Aggregated momentum optimizer.

    Defines an aggregated momentum optimizer (momentum with multiple
    timescales). Instead of a single learning rate and momentum mass,
    this optimizer includes `n` of them.

    Args:
      v0: Initial velocity, with shape (1,) or (n,). If it is a single
        number, this will be broadcast along each of the n modes.
      alphas: Learning rate hyperparameters with shape (n,).
      betas: Momentum hyperparameters with shape (n,).

    Returns:
      init_state: Initialize the optimizer state.
      update_opt: Updates the optimizer state variables given the current
        step, gradients, and current state.
      get_params: Gets parameters from the optimizer state.
    """
    alphas = jnp.reshape(alphas, (1, -1))
    betas = jnp.reshape(betas, (1, -1))

    def init_state(x):
      n = jnp.ravel(x).size
      v = jnp.ones((n, 1)) * jnp.reshape(v0, (1, -1))
      return (x, v)

    def update_opt(_, grads, state):
      x, v = state
      inputs = jnp.reshape(grads, (-1, 1))

      v_next = betas * v - alphas * inputs
      x_next = x + jnp.real(jnp.sum(v_next, axis=1))
      return (x_next, v_next)

    def get_params(state):
      return state[0]

    return (init_state, update_opt, get_params)

  return initial_meta_params, optimizer_fun
