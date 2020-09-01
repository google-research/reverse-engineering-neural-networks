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
"""Recurrent neural network (RNN) cells."""
import dataclasses

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node

__all__ = [
    'LinearRNN', 'RNNCell', 'StackedCell', 'GRU', 'LSTM', 'VanillaRNN', 'UGRNN',
    'embedding'
]

# Aliases for standard initializers and nonlinearities.
fan_in = jax.nn.initializers.variance_scaling(1., 'fan_in', 'normal')
zeros = jax.nn.initializers.zeros


@dataclasses.dataclass
class LinearRNN:
  """Dataclass for storing parameters of a Linear RNN."""
  A: jnp.array  # Input weights.      pylint: disable=invalid-name
  W: jnp.array  # Recurrent weights.  pylint: disable=invalid-name
  b: jnp.array  # Bias.

  def apply(self, x, h) -> jnp.array:
    """Linear RNN Update."""
    return self.A @ x + self.W @ h + self.b

  def flatten(self):
    return (self.A, self.W, self.b)


# Register the LinearRNN dataclass as a pytree, so that we can directly
# pass it to other jax functions (optimizers, flatten, etc.)
register_pytree_node(LinearRNN, lambda node: (node.flatten(), None),
                     lambda _, children: LinearRNN(*children))


class RNNCell:
  """Base class for all RNN Cells.

  An RNNCell must implement the following methods:
    init(PRNGKey, input_shape) -> output_shape, rnn_params
    apply(params, inputs, state) -> next_state
  """

  def __init__(self, num_units, h_init=zeros):
    """Initializes an RNNCell."""
    self.num_units = num_units
    self.h_init = h_init

    # Compute RNN Jacobians.
    self.inp_jac = jax.jacobian(self.apply, argnums=1)
    self.inp_jac.__doc__ = 'Computes the Jacobian of the RNN cell with respect to it\'s inputs.'  # pylint: disable=line-too-long

    self.rec_jac = jax.jacobian(self.apply, argnums=2)
    self.rec_jac.__doc__ = 'Computes the Jacobian of the RNN cell with respect to it\'s own hidden state.'  # pylint: disable=line-too-long

    # Generate cell update function that works on batches of inputs.
    self.batch_apply = jax.vmap(self.apply, in_axes=(None, 0, 0))
    self.batch_apply.__doc__ = 'Applies the RNN cell update to a batch of inputs and states.'  # pylint: disable=line-too-long

  def init(self, key, input_shape):
    raise NotImplementedError

  def apply(self, params, inputs, state):
    raise NotImplementedError

  def init_initial_state(self, key):
    return self.h_init(key, self.num_units)

  def get_initial_state(self, params, batch_size=None):
    """Gets initial RNN states.

    Args:
      params: rnn_parameters
      batch_size: batch size of initial states to create.

    Returns:
      An ndarray with shape (batch size, num_units).
    """
    if batch_size is None:
      return params['initial_state']
    else:
      initial_state = jnp.expand_dims(params['initial_state'], axis=0)
      return jnp.repeat(initial_state, batch_size, axis=0)


class StackedCell(RNNCell):
  """Stacks multiple RNN cells together.

  A stacked RNN cell is specified by a list of RNN cells and
  (optional) stax.Dense layers in between them.

  Note that the full hidden state for this cell is the concatenation
  of hidden states from all of the cells in the stack.
  """

  def __init__(self, layers):
    """Initializes a Stacked RNN Cell.

    Args:
      layers: list of layers, each of which is a (RNNCell, stax.Dense)
        tuple. The stax.Dense functions are used to transform the hidden
        state from the previous RNN in the stack to the inputs to the next
        RNN. If wish to feed the state in directly, use stax.Identity.
    """
    self.layers = layers

    # Total number of units across all cells in the stack.
    self.units = [c.num_units for c, _ in self.layers]
    super().__init__(sum(self.units))

  def init(self, key, input_shape):
    """Initializes parameters of a Stacked RNN."""

    shape = input_shape
    weights = []

    # Initialize each RNN cell and dense layer in the stack.
    for cell, (dense_init, _) in self.layers:
      key, cell_key, dense_key = jax.random.split(key, 3)
      cell_shape, cell_params = cell.init(cell_key, shape)
      shape, dense_params = dense_init(dense_key, cell_shape)
      weights.append((cell_params, dense_params))

    params = {
        'initial_state': super().init_initial_state(key),
        'weights': weights
    }

    return (input_shape[0], self.num_units), params

  def apply(self, params, inputs, state):
    """Applies a single step of a Stacked RNN."""

    # Split the RNN state vector into the state vector for each cell.
    states = jnp.split(state, jnp.cumsum(jnp.array(self.units))[:-1])
    next_states = []

    # Iteratively apply each RNN layer.
    for p, h, layer in zip(params['weights'], states, self.layers):

      # Get parameters and functions for this layer.
      cell_params, dense_params = p
      cell, (_, dense_apply) = layer

      # Apply the RNN and dense layers.
      next_state = cell.apply(cell_params, inputs, h)
      inputs = dense_apply(dense_params, h)

      # Store next state.
      next_states.append(next_state)

    return jnp.concatenate(next_states)


class VanillaRNN(RNNCell):
  """Vanilla RNN Cell."""

  def __init__(self, num_units, w_init=fan_in, b_init=zeros, h_init=zeros):
    """Vanilla RNN Cell.

    Args:
      num_units: int, Number of units in the RNN.
      w_init: Initializer for the recurrent & input weights (Default: fan_in).
      b_init: Initializer for the bias (Default: zeros).
      h_init: Initializer for the RNN initial condition (Default: zeros).
    """
    self.w_init = w_init
    self.b_init = b_init
    self.h_init = h_init
    super().__init__(num_units, h_init)

  def init(self, key, input_shape):
    """Initializes the parameters of a Vanilla RNN."""
    batch_size, num_inputs = input_shape

    rec_shape = (self.num_units, self.num_units)
    inp_shape = (self.num_units, num_inputs)
    bias_shape = (self.num_units,)
    output_shape = (batch_size, self.num_units)

    keys = jax.random.split(key, 4)
    weights = LinearRNN(self.w_init(keys[0], inp_shape),
                        self.w_init(keys[1], rec_shape),
                        self.b_init(keys[2], bias_shape))
    initial_state = super().init_initial_state(keys[3])
    params = {'initial_state': initial_state, 'weights': weights}
    return output_shape, params

  def apply(self, params, inputs, state):
    """Applies a single step of a Vanilla RNN."""
    linear_update = params['weights']
    return jnp.tanh(linear_update.apply(inputs, state))


class UGRNN(RNNCell):
  """Update-gate RNN Cell."""

  def __init__(self,
               num_units,
               gate_bias=0.,
               w_init=fan_in,
               b_init=zeros,
               h_init=zeros):
    """Update-gate RNN (UGRNN) Cell.

    Args:
      num_units: int, Number of units in the RNN.
      gate_bias: float, Bias to add to the update gate (Default: 0.).
      w_init: Initializer for the recurrent & input weights (Default: fan_in).
      b_init: Initializer for the bias (Default: zeros).
      h_init: Initializer for the RNN initial condition (Default: zeros).
    """
    self.gate_bias = gate_bias
    self.w_init = w_init
    self.b_init = b_init
    self.h_init = h_init
    super().__init__(num_units, h_init)

  def init(self, key, input_shape):
    batch_size, num_inputs = input_shape

    rec_shape = (self.num_units, self.num_units)
    inp_shape = (self.num_units, num_inputs)
    bias_shape = (self.num_units,)
    output_shape = (batch_size, self.num_units)

    def _build_linear_rnn(base_key):
      base_key, *keys = jax.random.split(base_key, 4)
      return base_key, LinearRNN(self.w_init(keys[0], inp_shape),
                                 self.w_init(keys[1], rec_shape),
                                 self.b_init(keys[2], bias_shape))

    # build internal gates / linear rnns.
    # we thread the `key` through the helper function `_build_linear_rnn`,
    # which returns a new key that we then pass to the next function call.
    key, update_gate = _build_linear_rnn(key)
    key, cell_state = _build_linear_rnn(key)

    key, key_ic = jax.random.split(key, 2)
    initial_state = super().init_initial_state(key_ic)
    params = {
        'initial_state': initial_state,
        'update_gate': update_gate,
        'cell_state': cell_state
    }
    return output_shape, params

  def apply(self, params, inputs, state):
    update_gate = params['update_gate']
    cell_state = params['cell_state']

    update = jax.nn.sigmoid(update_gate.apply(inputs, state) + self.gate_bias)
    cell = jnp.tanh(cell_state.apply(inputs, state))

    return update * state + (1 - update) * cell


class GRU(RNNCell):
  """Gated recurrent unit."""

  def __init__(self,
               num_units,
               gate_bias=0.,
               w_init=fan_in,
               b_init=zeros,
               h_init=zeros):
    """Gated recurrent unit (GRU) Cell.

    Args:
      num_units: int, Number of units in the RNN.
      gate_bias: float, Bias to add to the update gate (Default: 0.).
      w_init: Initializer for the recurrent & input weights (Default: fan_in).
      b_init: Initializer for the bias (Default: zeros).
      h_init: Initializer for the RNN initial condition (Default: zeros).
    """
    self.gate_bias = gate_bias
    self.w_init = w_init
    self.b_init = b_init
    self.h_init = h_init
    super().__init__(num_units, h_init)

  def init(self, key, input_shape):
    batch_size, num_inputs = input_shape

    rec_shape = (self.num_units, self.num_units)
    inp_shape = (self.num_units, num_inputs)
    bias_shape = (self.num_units,)
    output_shape = (batch_size, self.num_units)

    def _build_linear_rnn(base_key):
      base_key, *keys = jax.random.split(base_key, 4)
      return base_key, LinearRNN(self.w_init(keys[0], inp_shape),
                                 self.w_init(keys[1], rec_shape),
                                 self.b_init(keys[2], bias_shape))

    # build internal gates / linear rnns.
    # we thread the `key` through the helper function `_build_linear_rnn`,
    # which returns a new key that we then pass to the next function call.
    key, update_gate = _build_linear_rnn(key)
    key, reset_gate = _build_linear_rnn(key)
    key, cell_state = _build_linear_rnn(key)

    key, key_ic = jax.random.split(key, 2)
    initial_state = super().init_initial_state(key_ic)
    params = {
        'initial_state': initial_state,
        'update_gate': update_gate,
        'reset_gate': reset_gate,
        'cell_state': cell_state
    }
    return output_shape, params

  def apply(self, params, inputs, state):
    update_gate = params['update_gate']
    reset_gate = params['reset_gate']
    cell_state = params['cell_state']

    update = jax.nn.sigmoid(update_gate.apply(inputs, state) + self.gate_bias)
    reset = jax.nn.sigmoid(reset_gate.apply(inputs, state))
    cell = jnp.tanh(cell_state.apply(inputs, reset * state))

    return update * state + (1 - update) * cell


class LSTM(RNNCell):
  """Long-short term memory (LSTM)."""

  def __init__(self,
               num_units,
               forget_bias=1.,
               w_init=fan_in,
               b_init=zeros,
               h_init=zeros):
    """Long-short term memory (LSTM) Cell.

    Args:
      num_units: int, Number of units in the RNN.
      forget_bias: float, Bias to add to the forget gate. (Default: 1.0).
      w_init: Initializer for the recurrent & input weights (Default: fan_in).
      b_init: Initializer for the bias (Default: zeros).
      h_init: Initializer for the RNN initial condition (Default: zeros).
    """
    self.forget_bias = forget_bias
    self.w_init = w_init
    self.b_init = b_init
    self.h_init = h_init

    # Note that we double the number of units here, since the LSTM
    # contains units for both the hidden and cell states.
    # These are h(t) and c(t) in most LSTM equations.
    super().__init__(num_units * 2, h_init)

  def init(self, key, input_shape):
    batch_size, num_inputs = input_shape

    # We divide by 2 since for the LSTM, there are separate
    # updates for the cell and hidden state. The full state
    # will be split in two in the apply() method.
    n = int(self.num_units / 2)
    rec_shape = (n, n)
    inp_shape = (n, num_inputs)
    bias_shape = (n,)

    # The output contains the full number of units.
    output_shape = (batch_size, self.num_units)

    def _build_linear_rnn(key):
      key, *prngs = jax.random.split(key, 4)
      return key, LinearRNN(self.w_init(prngs[0], inp_shape),
                            self.w_init(prngs[1], rec_shape),
                            self.b_init(prngs[2], bias_shape))

    # Build internal gates / linear RNNs.
    # We thread the `key` through the helper function `_build_linear_rnn`,
    # which returns a new key that we then pass to the next function call.
    key, forget_gate = _build_linear_rnn(key)
    key, update_gate = _build_linear_rnn(key)
    key, output_gate = _build_linear_rnn(key)
    key, cell_state = _build_linear_rnn(key)

    key, key_ic = jax.random.split(key, 2)
    initial_state = super().init_initial_state(key_ic)
    params = {
        'initial_state': initial_state,
        'forget_gate': forget_gate,
        'update_gate': update_gate,
        'output_gate': output_gate,
        'cell_state': cell_state
    }
    return output_shape, params

  def apply(self, params, inputs, full_state):
    forget_gate = params['forget_gate']
    update_gate = params['update_gate']
    output_gate = params['output_gate']
    cell_state = params['cell_state']

    # Split full state into the hidden and cell states.
    state, cell = jnp.split(full_state, 2, axis=-1)

    f = jax.nn.sigmoid(forget_gate.apply(inputs, state) + self.forget_bias)
    i = jax.nn.sigmoid(update_gate.apply(inputs, state))
    o = jax.nn.sigmoid(output_gate.apply(inputs, state))
    c = jnp.tanh(cell_state.apply(inputs, state))

    new_cell = f * cell + i * c
    new_state = o * jnp.tanh(new_cell)

    return jnp.concatenate((new_state, new_cell), axis=-1)


def embedding(vocab_size,
              embedding_size,
              initializer=jax.nn.initializers.orthogonal()):
  """Builds a token embedding.

  Args:
    vocab_size: int, Size of the vocabulary (number of tokens).
    embedding_size: int, Dimensionality of the embedding.
    initializer: Initializer for the embedding (Default: orthogonal).

  Returns:
    init_fun: callable, Initializes the embedding given a key and input_shape.
    apply_fun: callable, Converts a set of tokens to embedded vectors.
  """

  def init_fun(key, input_shape):
    """Build an embedding matrix."""
    shape = (vocab_size, embedding_size)
    emb = initializer(key, shape)
    output_shape = input_shape + (embedding_size,)
    return output_shape, emb

  def apply_fun(emb, tokens, **kwargs):
    """Embedding lookup."""
    del kwargs  # unused
    return jnp.take(emb, tokens, axis=0)

  return init_fun, apply_fun
