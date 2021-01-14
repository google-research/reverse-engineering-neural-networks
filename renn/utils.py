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
"""Utilities for optimization."""

import jax
from jax import flatten_util
import jax.numpy as jnp

import numpy as np
import tqdm

__all__ = [
    'batch_mean', 'norm', 'identity', 'fst', 'snd', 'optimize', 'one_hot',
    'compose'
]


def batch_mean(fun, in_axes):
  """Converts a function to a batched version (maps over multiple inputs).

  This takes a function that returns a scalar (such as a loss function) and
  returns a new function that maps the function over multiple arguments (such
  as over multiple random seeds) and returns the average of the results.

  It is useful for generating a batched version of a loss function, where the
  loss function has stochasticity that depends on a random seed argument.

  Args:
    fun: function, Function to batch.
    in_axes: tuple, Specifies the arguments to fun to batch over. For
      example, in_axes=(None, 0) would batch over the second argument.

  Returns:
    batch_fun: function, computes the average over a batch.
  """
  mapped_fun = jax.vmap(fun, in_axes=in_axes)

  def batch_fun(*args):
    return jnp.mean(mapped_fun(*args))

  return batch_fun


def norm(params, order=2):
  """Computes the (flattened) norm of a pytree."""
  return jnp.linalg.norm(flatten_util.ravel_pytree(params)[0], ord=order)


def identity(x):
  """Identity function."""
  return x


def fst(xs):
  """Returns the first element from a list."""
  return xs[0]


def snd(xs):
  """Returns the second element from a list."""
  return xs[1]


def compose(*funcs):
  """Returns a function that is the composition of multiple functions."""

  def wrapper(x):
    for func in reversed(funcs):
      x = func(x)
    return x

  return wrapper


def build_mask(max_length: int):
  """Builds a function that generates a binary mask.

  For example, `f = build_mask(5)` returns a function that generates masks of
  total length 5. Calling this function with an array of integers, e.g.
  f(jnp.array([2, 3])), will return a binary (mask) array:
    [[1., 1., 0., 0., 0.],
     [1., 1., 1., 0., 0.]]

  This is useful for generating binary arrays for masking out elements of a
  padded batch, such as computing a loss over a batch of padded sequences.

  Args:
    max_length: int, The total length of the mask/sequence.

  Returns:
    mask_fun: function, Takes an array of indices (lengths) and returns
      a binary mask where
  """

  def mask_fun(index: jnp.array) -> jnp.array:
    """Builds a binary mask."""
    return jnp.where(
        jnp.arange(max_length) < index, jnp.ones(max_length),
        jnp.zeros(max_length))

  return jax.vmap(mask_fun)


def select(sequences, indices):
  """Selects a particular timestep from a batch of sequences."""
  last_index = jnp.array(indices)[:, jnp.newaxis, jnp.newaxis]
  return jnp.squeeze(jnp.take_along_axis(sequences, last_index, axis=1))


def optimize(loss_fun, x0, optimizer, steps, stop_tol=-np.inf):
  """Run an optimizer on a given loss function.

  Args:
    loss_fun: Scalar loss function to optimize.
    x0: Initial parameters.
    optimizer: An tuple of optimizer functions (init_opt, update_opt,
      get_params) from a jax.experimental.optimizers instance.
    steps: Iterator over steps.
    stop_tol: Stop if the loss is below this value (Default: -np.inf).

  Returns:
    loss_hist: Array of losses during training.
    final_params: Optimized parameters.
  """

  # Initialize optimizer.
  init_opt, update_opt, get_params = optimizer
  opt_state = init_opt(x0)

  # Loss and gradient.
  value_and_grad = jax.value_and_grad(loss_fun)

  @jax.jit
  def step(k, state):
    params = get_params(state)
    loss, grads = value_and_grad(params)
    return loss, update_opt(k, grads, state)

  # Store loss history.
  loss_hist = []
  for k in steps:
    f, opt_state = step(k, opt_state)
    loss_hist.append(f)

    if f <= stop_tol:
      break

  # Extract final parameters.
  final_params = get_params(opt_state)

  return np.array(loss_hist), final_params


def one_hot(labels, num_classes, dtype=jnp.float32):
  """Creates a one-hot encoding of an array of labels.

  Args:
    labels: array of integers with shape (num_examples,).
    num_classes: int, Total number of classes.
    dtype: optional, jax datatype for the return array (Default: float32).

  Returns:
    one_hot_labels: array with shape (num_examples, num_classes).
  """
  return jnp.array(jnp.array(labels)[:, None] == jnp.arange(num_classes), dtype)


def select(sequences, indices):
  """Given an array of shape (number_of_sequences, sequence_length, element_dimension),
  and a 1D array specifying which indices of each sequence to select, return
  a (number_of_sequences, element_dimension)-shaped array with the selected elements.

  Args:
    sequences: array with shape (number_of_sequences, sequence_length, element_dimension)
    indices: 1D array with length number_of_sequence

  Returns:
    selected_elements: array with shape (number_of_sequences, element_dimension)
  """

  assert len(indices) == sequences.shape[0]

  # shape indices properly
  indices_shaped = indices[:, jnp.newaxis, jnp.newaxis]

  # select element
  selected_elements = jnp.take_along_axis(sequences, indices_shaped, axis=1)

  # remove sequence dimension
  selected_elements = jnp.squeeze(selected_elements, axis=1)

  return selected_elements


def make_loss_function(network_apply_fun, basic_loss_fun, regularization_fun):
  """ Given the network-function, the basic loss function, and
  a regularization function, return a loss function which maps a tuple of
  network parameters and a training batch to a loss value.

  Arguments:
    network_apply_fun - maps (network_params, batched_inputs) -> network_logits
    basic_loss_fun - maps (logits, batched_labels) -> scalar loss value
    regularization_fun - maps network_params -> scalar loss value

  Returns:
    total_loss_fun - maps (network_params, batch) -> scalar loss value
  """

  def total_loss_fun(params, batch):
    """
    Maps network parameters and training batch to a loss value.

    Args:
      batch: a dictionary with keys ['inputs', 'index', 'labels']
        'inputs': sequence of inputs with shape (batch_size, max_sequence_length)
        'index' : 1d-array storing length of the corresponding input sequence
        'labels': 1d-array storing label of corresponding input sequence

    Returns:
      loss: scalar loss averaged over batch
    """

    all_time_logits = network_apply_fun(params, batch['inputs'])
    end_logits = select(all_time_logits, batch['index'] - 1)

    return basic_loss_fun(end_logits,
                          batch['labels']) + regularization_fun(params)

  return total_loss_fun


def make_acc_fun(network_apply_fun, num_outputs=1):
  """ Given a network function and number of outputs, returns an accuracy
  function """

  if num_outputs == 1:
    prediction_function = lambda x: (x >= 0.).astype(jnp.int32)
  else:
    prediction_function = lambda x: x.argmax(axis=-1).astype(jnp.int32)

  @jax.jit
  def accuracy_fun(params, batch):
    all_time_logits = network_apply_fun(params, batch['inputs'])
    end_logits = select(all_time_logits, batch['index'] - 1)
    predictions = jnp.squeeze(prediction_function(end_logits))
    accuracies = (batch['labels'] == predictions).astype(jnp.int32)
    return jnp.mean(accuracies)

  return accuracy_fun
