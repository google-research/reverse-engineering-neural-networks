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
"""Neural networks classification loss on mnist."""

import jax
from jax.experimental import stax
import jax.numpy as jnp

import numpy as np
import tensorflow_datasets as tfds

from ... import utils


def load_mnist():
  """Download and normalize the MNIST dataset."""
  # Fetch full datasets for evaluation
  mnist_data = tfds.load(name='mnist', batch_size=-1)
  mnist_data = tfds.as_numpy(mnist_data)

  datasets = {}
  for key in ('train', 'test'):
    data = mnist_data[key]
    images, labels = data['image'], data['label']
    datasets[key] = (np.array(images), np.array(labels))

  return datasets


def fully_connected(num_classes, layers=(64, 64)):
  """Build a fully connected neural network."""
  stack = [stax.Flatten]

  # Concatenate fully connected layers.
  for num_units in layers:
    stack += [stax.Dense(num_units), stax.Relu]

  # Output layer.
  stack += [stax.Dense(num_classes), stax.LogSoftmax]

  return stax.serial(*stack)


def conv2d(num_classes, layers=((32, 5, 2), (16, 3, 2), (16, 3, 2))):
  """Builds a simple convolutional neural network."""
  stack = []

  # Concatenate convolutional layers.
  for num_units, kernel_size, stride in layers:
    stack += [
        stax.Conv(num_units, (kernel_size, kernel_size), (stride, stride),
                  padding='SAME'), stax.Relu
    ]

  # Output layer.
  stack += [stax.Flatten, stax.Dense(num_classes), stax.LogSoftmax]

  return stax.serial(*stack)


def classifier(model, features, targets, batch_size=32, num_classes=10):
  """Builds a classification task using a fully connected network.

  Args:
    model: tuple, A jax.experimental.stax function pair.
    features: array with shape (num_examples, *feature_dimensions)
    targets: array with shape (num_examples, num_classes)
    batch_size: int, Number of examples to use per batch (Default: 32).
    num_classes: int, Number of output classes.

  Returns:
    problem_fun: function that takes a PRNGKey and returns initial
      parameters and a loss function.
  """

  # Prepare data.
  input_shape = (-1,) + features.shape[1:]
  num_examples = features.shape[0]

  # Model.
  init_fun, predict_fun = model

  def problem_fun(prng_key):
    """Builds a problem given a PRNG key.

    Args:
      prng_key: jax.random PRNG key.

    Returns:
      initial_params: pytree containing initial parameters.
      loss_fun: function that takes (params, step) and returns a scalar loss.
    """
    params_key, data_key = jax.random.split(prng_key)

    # Generate initial parameters for this problem.
    _, initial_params = init_fun(params_key, input_shape)

    # Classification loss function.
    def loss_fun(params, step):
      """Computes the cross-entropy classification loss."""

      # Subselect a minibatch of data.
      key = jax.random.fold_in(data_key, step)
      indices = jax.random.permutation(key, jnp.arange(num_examples))
      batch_indices = indices[:batch_size]
      batch_features = features[batch_indices]
      batch_targets = targets[batch_indices]

      # Preprocess features and targets.
      batch_features = batch_features.astype(jnp.float32) / 255.
      batch_targets = utils.one_hot(batch_targets, num_classes)

      # Evaluate loss.
      outputs = predict_fun(params, batch_features)
      return -jnp.mean(jnp.sum(outputs * batch_targets, axis=1))

    return initial_params, loss_fun

  return problem_fun
