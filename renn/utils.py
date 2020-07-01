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
    'batch_mean',
    'norm',
    'identity',
    'fst',
    'snd',
    'optimize',
    'one_hot'
]


def batch_mean(fun, in_axes):
  """Converts a scalar function to a batched version (maps over multiple inputs).

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


def optimize(loss_fun, x0, optimizer, num_steps,
             iterator=tqdm.trange, stop_tol=-np.inf):
  """Run an optimizer on a given loss function.

  Args:
    loss_fun: Scalar loss function to optimize.
    x0: Initial parameters.
    optimizer: An tuple of optimizer functions (init_opt, update_opt,
      get_params) from a jax.experimental.optimizers instance.
    num_steps: Number of steps to run the optimizer.
    iterator: Function used to iterate over steps (Default: tqdm.trange).
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
  loss_hist = np.empty(num_steps)
  for k in iterator(num_steps):
    f, opt_state = step(k, opt_state)
    loss_hist[k] = f

    if f <= stop_tol:
      loss_hist = loss_hist[:k]
      break

  # Extract final parameters.
  final_params = get_params(opt_state)

  return loss_hist, final_params


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
