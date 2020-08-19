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
"""Recurrent neural network (RNN) helper functions."""

import functools

import jax
from jax.experimental import stax
import jax.numpy as jnp
import numpy as np

from . import cells
from . import unroll

__all__ = ['build_rnn', 'mse', 'eigsorted', 'timescale']


def build_rnn(num_tokens, emb_size, cell, num_outputs=1):
  """Builds an end-to-end recurrent neural network (RNN) model.

  Args:
    num_tokens: int, Number of different input tokens.
    emb_size: int, Dimensionality of the embedding vectors.
    cell: RNNCell to use as the core update function (see cells.py).
    num_outputs: int, Number of outputs from the readout (Default: 1).

  Returns:
    init_fun: function that takes a PRNGkey and input_shape and returns
      expected shapes and initialized embedding, RNN, and readout parameters.
    apply_fun: function that takes a tuple of network parameters and batch of
      input tokens and applies the RNN to each sequence in the batch.
    emb_apply: function to just apply the embedding.
    readout_apply: function to just apply the readout.
  """
  emb_init, emb_apply = cells.embedding(num_tokens, emb_size)
  readout_init, readout_apply = stax.Dense(num_outputs)

  def init_fun(key, input_shape):
    """Initialize the components of the RNN.

    Args:
      key: Jax PRNGkey used to initialize the parameters.
      input_shape: tuple representing the input shape, should be
        (batch_size, sequence_length).

    Returns:
      shapes: set of tuples representing the shapes after applying
        the Embedding, RNN Cell, and Readout layers.
      network_params: tuple of network parameters, containing the
        embedding, RNN cell, and readout parameters.
    """

    emb_key, cell_key, readout_key = jax.random.split(key, 3)

    # Initialize the Embedding for the input tokens.
    emb_shape, emb_params = emb_init(emb_key, input_shape)

    # The cell is defined for a single update step, which is why we ignore
    # the sequence dimension (emb_shape[1]) here.
    rnn_shape, rnn_params = cell.init(cell_key, (emb_shape[0], emb_shape[2]))

    output_shape, readout_params = readout_init(readout_key, rnn_shape)

    shapes = (emb_shape, rnn_shape, output_shape)
    network_params = (emb_params, rnn_params, readout_params)

    return shapes, network_params

  def apply_fun(network_params, tokens):
    """Applies the RNN on a batch of input sequences.

    Args:
      network_params: tuple of network parameters (see init_fun).
      tokens: batch of inputs, with shape (batch_size, sequence_length).

    Returns:
      outputs: network outputs, at every step along the sequence.
    """
    emb_params, rnn_params, readout_params = network_params

    # Apply the embedding.
    inputs = emb_apply(emb_params, tokens)

    # Run the RNN.
    initial_states = cell.get_initial_state(rnn_params,
                                            batch_size=tokens.shape[0])
    return unroll.unroll_rnn(initial_states, inputs,
                             functools.partial(cell.batch_apply, rnn_params),
                             functools.partial(readout_apply, readout_params))

  return init_fun, apply_fun, emb_apply, readout_apply


def mse(y, yhat):
  """Mean squared error loss."""
  return 0.5 * jnp.mean((y - yhat)**2)


def eigsorted(jac):
  """Computes sorted eigenvalues and corresponding eigenvectors of a matrix.

  Notes:
    The eigenvectors are stored in the columns of the returned matrices.
    The right and left eigenvectors are returned, such that: J=REL^T

  Args:
    jac: numpy array used to compute the eigendecomposition (must be square).

  Returns:
    rights: right eigenvectors, as columns in the returned array.
    eigvals: numpy array of eigenvalues.
    lefts: left eigenvectors, as columns in the returned array.
  """
  unsorted_eigvals, unsorted_rights = np.linalg.eig(jac)
  sorted_indices = np.flipud(np.argsort(np.abs(unsorted_eigvals)))

  eigenvalues = unsorted_eigvals[sorted_indices]
  rights = unsorted_rights[:, sorted_indices]
  lefts = np.linalg.pinv(rights).T

  return rights, eigenvalues, lefts


def timescale(eigenvalues):
  """Converts eigenvalues into approximate time constants."""
  return -1. / np.log(np.abs(eigenvalues))
