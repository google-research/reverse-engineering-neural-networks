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
"""Unroll functions for recurrent neural network (RNN) cells."""

import jax
import jax.numpy as jnp

from renn.utils import identity

__all__ = ['unroll_rnn', 'unroll_reinject']


def unroll_rnn(initial_states,
               input_sequences,
               apply_rnn,
               apply_readout=identity):
  """Unrolls an RNN on a batch of input sequences.

  Given a batch of initial RNN states, and a batch of input sequences, this
  function unrolls application of the RNN along the sequence. The RNN state
  is updated using the `apply_rnn` function, and the `readout` is used to
  convert the RNN state to outputs (defaults to the identity function).

  B: batch size.
  N: number of RNN units.
  T: sequence length.

  Args:
    initial_states: batch of initial states, with shape (B, N).
    input_sequences: batch of inputs, with shape (B, T, N).
    apply_rnn: updates the RNN hidden state, given (inputs, current_states).
    readout: applies the readout, given current states. If this is the identity
      function, then no readout is applied (returns the hidden states).

  Returns:
    outputs: batch of outputs (batch_size, sequence_length, num_outputs).
  """

  def _step(state, inputs):
    next_state = apply_rnn(inputs, state)
    outputs = apply_readout(next_state)
    return next_state, outputs

  input_sequences = jnp.swapaxes(input_sequences, 0, 1)
  _, outputs = jax.lax.scan(_step, initial_states, input_sequences)

  return jnp.swapaxes(outputs, 0, 1)


def unroll_reinject(initial_states, initial_token, sequence_length,
                    apply_embedding, apply_rnn, apply_readout):
  """Unrolls an RNN, reinjecting the output back into the RNN."""

  def _step(state, _):

    # Unpack loop state.
    tokens, rnn_state = state

    # Apply embedding, RNN, and readout.
    rnn_inputs = apply_embedding(tokens)
    rnn_state = apply_rnn(rnn_inputs, rnn_state)
    logits = apply_readout(rnn_state)

    # Pack new loop state
    next_state = (jnp.argmax(logits, axis=-1), rnn_state)

    return next_state, logits

  # Format scan arguments.
  batch_size = initial_states.shape[0]
  batch_inputs = initial_token * jnp.ones(batch_size).astype(jnp.int32)
  dummy_inputs = jnp.zeros((sequence_length, 1))

  # Unroll loop via scan.
  _, outputs = jax.lax.scan(_step, (batch_inputs, initial_states), dummy_inputs)

  return jnp.swapaxes(outputs, 0, 1)
