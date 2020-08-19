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
"""Functions for computing loss."""

import jax.numpy as jnp
from jax import lax

__all__ = ['binary_xent', 'multiclass_xent']


def binary_xent(logits, labels):
  """ Cross-entropy loss in in a two-class classification problem,
  where the model output is a single logit

  Args:
    logits: array of shape (batch_size, 1) or just (batch_size)
    labels: array of length batch_size, whose elements are either 0 or 1

  Returns:
    loss: scalar cross entropy loss
  """

  squeezed_logits = jnp.squeeze(logits)
  log_likelihood = jnp.maximum(squeezed_logits, 0) - squeezed_logits * labels + \
      jnp.log(1 + jnp.exp(-jnp.abs(squeezed_logits)))
  return jnp.mean(log_likelihood)


def multiclass_xent(logits, labels):
  # zero max of logit
  shifted = logits - lax.stop_gradient(logits.max(axis=-1, keepdims=True))
  log_probs = shifted - jnp.log(
      jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))

  log_likelihood = jnp.take_along_axis(log_probs,
                                       labels[:, jnp.newaxis],
                                       axis=1)
  xent_loss = -1 * jnp.mean(log_likelihood)

  return xent_loss
