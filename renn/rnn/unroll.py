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

import jax
import jax.numpy as jnp

__all__ = ['unroll_rnn']


def identity(x):
  """Identity function f(x) = x."""
  return x


def unroll_rnn(initial_states, input_sequences, rnn_update, readout=identity):
  """Unrolls an RNN on a batch of input sequences.

  Given a batch of initial RNN states, and a batch of input sequences, this
  function unrolls application of the RNN along the sequence. The RNN state
  is updated using the `rnn_update` function, and the `readout` is used to
  convert the RNN state to outputs (defaults to the identity function).

  B: batch size.
  N: number of RNN units.
  T: sequence length.

  Args:
    initial_states: batch of initial states, with shape (B, N).
    input_sequences: batch of inputs, with shape (B, T, N).
    rnn_update: updates the RNN hidden state, given (inputs, current_states).
    readout: applies the readout, given current states. If this is the identity
      function, then no readout is applied (returns the hidden states).

  Returns:
    outputs: batch of outputs (batch_size, sequence_length, num_outputs).
  """

  def _step(state, inputs):
    next_state = rnn_update(inputs, state)
    outputs = readout(next_state)
    return next_state, outputs

  input_sequences = jnp.swapaxes(input_sequences, 0, 1)
  _, outputs = jax.lax.scan(_step, initial_states, input_sequences)

  return jnp.swapaxes(outputs, 0, 1)
