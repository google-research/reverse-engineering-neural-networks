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
"""Classes for attention"""
import dataclasses

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node

import numpy as np

__all__ = ['dot_product_attention']

def dot_product_attention(queries: jnp.array,
                          keys: jnp.array,
                          values: jnp.array) -> jnp.array:
  """Computes dot-product attention.

  Arguments:
    queries: jnp.array, of shape (n_tokens, query_dimension)
    keys:    jnp.array, of shape (n_tokens, key_dimension)
    values:  jnp.array, of shape (n_tokens, value_dimension)

  key_dimension must be equal to query_dimension
  """

  dot_products = jnp.dot(queries, keys.T) #A_ij = q_i . key_j
  attention_weights = jax.nn.softmax(dot_products, axis=-1)

  return jnp.dot(attention_weights, values)
