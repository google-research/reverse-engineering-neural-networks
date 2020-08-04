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

"""Tests RNN cells."""

import jax
import jax.numpy as jnp
import numpy as np

import pytest

from renn.rnn import cells


class TestLinearRNN:
  """Tests the Linear RNN dataclass."""

  def setup_method(self):

    # Define weights of the RNN cell.
    self.num_units = 5
    self.num_inputs = 1

    self.rec_weights = jnp.eye(self.num_units)
    self.inp_weights = jnp.zeros((self.num_units, self.num_inputs))
    self.bias = jnp.zeros(self.num_units,)

    # Build a Linear RNN cell using these weights.
    self.rnn = cells.LinearRNN(self.inp_weights, self.rec_weights, self.bias)

  def test_fields(self):
    """Tests fields of the Linear RNN cell."""
    assert self.rnn.A is self.inp_weights
    assert self.rnn.W is self.rec_weights
    assert self.rnn.b is self.bias

  def test_apply(self):
    """Tests the apply() method."""
    inputs = jnp.array([1.])
    state = jnp.arange(self.num_units).astype(jnp.float32)
    new_state = self.rnn.apply(inputs, state)

    # For this RNN, the new state should equal the previous state.
    assert np.allclose(state, new_state)


def test_embedding():
  """Tests the embedding (discrete lookup) layer."""

  # Setup.
  key = jax.random.PRNGKey(0)
  vocab_size = 100
  emb_size = 32
  initializer = jax.nn.initializers.orthogonal()

  # Build embedding layer.
  init_fun, apply_fun = cells.embedding(vocab_size, emb_size, initializer)

  # Initialize embedding.
  input_shape = (-1,)
  output_shape, emb = init_fun(key, input_shape)

  # Embeddings should be orthogonal.
  assert np.allclose(emb.T @ emb, jnp.eye(emb_size), atol=1e-5)

  # Output shape is the input shape with an additional embedding dimension.
  assert output_shape == input_shape + (emb_size,)

  # Apply the embedding to each possible token.
  tokens = jnp.arange(vocab_size)
  outputs = apply_fun(emb, tokens)
  assert np.allclose(outputs, emb)


@pytest.fixture(scope="module", params=[cells.VanillaRNN, cells.GRU, cells.LSTM, cells.UGRNN])
def rnn_cell(request):
  """Fixture that builds an RNN Cell."""
  base_key = jax.random.PRNGKey(0)
  keys = jax.random.split(base_key, 3)

  num_units = 32
  num_inputs = 16

  # Build RNN Cell.
  cell = request.param(num_units)
  input_shape = (-1, num_inputs)
  output_shape, params = cell.init(keys[0], input_shape)

  # Generate random hidden state and inputs.
  state = jax.random.normal(keys[1], (cell.num_units,))
  inputs = jax.random.normal(keys[2], (num_inputs,))

  return (cell, output_shape, params, inputs, state)


def test_rnn_output_shape(rnn_cell):
  cell, output_shape, params, inputs, state = rnn_cell
  expected_output_shape = (-1, cell.num_units)
  assert output_shape == expected_output_shape


def test_rnn_apply(rnn_cell):
    """Tests the single step apply method."""
    cell, output_shape, params, inputs, state = rnn_cell

    new_state = cell.apply(params, inputs, state)

    # State shape should not change.
    assert new_state.shape == state.shape


def test_batch_initial_shape(rnn_cell):
  """Tests method to get a batch of initial states."""
  cell, output_shape, params, inputs, state = rnn_cell
  bz = 256

  # Generate a batch of initial states.
  states = cell.get_initial_state(params, batch_size=(bz,))

  # Test shape.
  assert states.shape == (bz, cell.num_units)

  # Initial states across the batch should be the same.
  assert np.allclose(np.diff(states, axis=0), 0.)


def test_input_jacobian(rnn_cell):
  """Tests the ability to compute Jacobians wrt. the inputs."""
  cell, output_shape, params, inputs, state = rnn_cell
  jac = cell.inp_jac(params, inputs, state)
  assert jac.shape == (cell.num_units, inputs.size)


def test_recurrent_jacobian(rnn_cell):
  """Tests the ability to compute Jacobians wrt. the recurrent state."""
  cell, output_shape, params, inputs, state = rnn_cell
  jac = cell.rec_jac(params, inputs, state)
  assert jac.shape == (cell.num_units, cell.num_units)


def test_batch_apply(rnn_cell):
  """Tests the ability to apply the RNN to a batch of inputs."""
  cell, output_shape, params, inputs, state = rnn_cell
  batch_size = 256

  # Generate batch inputs and states.
  batch_inputs = jnp.repeat(inputs[jnp.newaxis, :], batch_size, axis=0)
  batch_states = jnp.repeat(state[jnp.newaxis, :], batch_size, axis=0)

  # Apply the RNN.
  new_states = cell.batch_apply(params, batch_inputs, batch_states)

  # Test shape.
  assert new_states.shape == (batch_size, cell.num_units)


def test_vrnn_param_shape():
  """Tests parameter shape of the Vanilla RNN class."""

  # Build Vanilla RNN cell.
  num_inputs = 16
  num_units = 32
  cell = cells.VanillaRNN(num_units)

  # Initialize parameters.
  key = jax.random.PRNGKey(0)
  _, params = cell.init(key, (-1, num_inputs))

  expected_shape = {
      'initial_state': (num_units,),
      'weights': (
          (num_units, num_inputs),  # Input weight matrix.
          (num_units, num_units),   # Recurrent weight matrix.
          (num_units,))                  # Bias shape.
  }

  # Test shape of the initial state vector.
  assert params['initial_state'].shape == expected_shape['initial_state']

  # Test shape of the RNN cell weights.
  for weight, expected_weight_shape in zip(params['weights'].flatten(),
                                           expected_shape['weights']):
    assert weight.shape == expected_weight_shape
