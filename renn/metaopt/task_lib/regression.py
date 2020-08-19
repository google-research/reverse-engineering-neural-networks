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
"""Defines regression tasks."""

import jax
import jax.numpy as jnp


def linear(dim, num_datapoints, scale, l2_penalty=1e-3):
  """Generates a random linear regression problem."""

  def problem_fun(prng_key):
    """Builds a problem given a PRNG key."""
    # Split the PRNG into two keys.
    params_key, features_key, targets_key = jax.random.split(prng_key, 3)

    # Generate a random feature matrix and target vector.
    features = scale * jax.random.normal(features_key, (num_datapoints, dim))
    targets = scale * jax.random.normal(targets_key, (num_datapoints,))

    # Generate initial parameters for this problem.
    initial_params = jax.random.normal(params_key, (dim,))

    # Quadratic loss function.
    def loss_fun(params, _):
      predictions = jnp.dot(features, params)
      loss = jnp.mean((predictions - targets)**2)
      return loss + l2_penalty * jnp.sum(params**2)

    return initial_params, loss_fun

  return problem_fun


def logistic(dim, num_datapoints, l2_penalty=1e-5):
  """Generates a random logistic regression problem."""

  def problem_fun(prng_key):
    """Builds a problem given a PRNG key."""

    # Split the PRNG into two keys.
    params_key, data_key = jax.random.split(prng_key, 2)
    features, targets = make_classification(data_key, dim, num_datapoints)

    # Generate initial parameters for this problem.
    initial_params = {
        'weights': jax.random.normal(params_key, (dim,)),
        'bias': jnp.zeros(1),
    }

    def loss_fun(params, _):
      """Logistic loss function."""
      logits = jnp.dot(features, params['weights']) + params['bias']

      data_loss = sigmoid_xent(logits, targets)
      reg_loss = l2_penalty * jnp.sum(params['weights']**2)
      return data_loss + reg_loss

    return initial_params, loss_fun

  return problem_fun


def sigmoid_xent(logits, labels):
  """Sigmoid cross-entropy (used for logistic regression)."""
  return jnp.mean(jnp.log(1 + jnp.exp(-logits)) + logits * (1 - labels))


def logistic_acc(logits, labels):
  """Logistic regression accuracy."""
  predictions = logits > 0.
  return jnp.mean(predictions == labels)


def make_classification(prng_key,
                        num_features,
                        num_examples,
                        num_classes=2,
                        sep=2.):
  """Generates random classification problems."""

  num_examples_per_class = num_examples // num_classes
  features, labels = [], []

  for label in range(num_classes):

    # Class mean is on a vertex of the hypercube.
    # e.g. mu0 = [sep,   0, 0, 0]
    #      mu1 = [  0, sep, 0, 0]
    # and so on.
    mu = sep * jnp.eye(num_features)[:, label]

    # Sample data from a multivariate normal centered at the class mean.
    keys = jax.random.split(prng_key, 2)
    sqrt_sigma = jax.random.normal(keys[0], (num_features, num_features))
    samples = jax.random.normal(keys[1], (num_examples_per_class, num_features))
    features.append(jnp.atleast_2d(mu) + jnp.dot(samples, sqrt_sigma))
    labels.append(label * jnp.ones(num_examples_per_class))

  return jnp.vstack(features), jnp.hstack(labels)
