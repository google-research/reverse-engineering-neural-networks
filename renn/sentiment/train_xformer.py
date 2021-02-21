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

"""Sentiment classification using Transformer."""

import functools

from absl import app
from absl import flags
from absl import logging

from flax import jax_utils
from flax import nn
from flax.training import checkpoints
from flax.training import common_utils

import jax
from jax import random
import jax.numpy as jnp
import tensorflow_datasets as tfds
import numpy as np

from renn.sentiment import data
from renn.transformer import build_model

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', default=32, help='Batch size for training.')

flags.DEFINE_string(
    'model_dir', default=None, help='Directory to store model data.')

flags.DEFINE_integer('eval_freq', default=1000, help='Evaluation frequency.')

flags.DEFINE_integer(
    'num_train_steps', default=500000, help='Number of training steps.')

flags.DEFINE_float('learning_rate', default=0.05, help='Learning rate.')

flags.DEFINE_float(
    'weight_decay',
    default=1e-1,
    help='Decay factor for AdamW-style weight decay.')

flags.DEFINE_float(
    'dropout_rate', default=0.1, help='Dropout probability.')

flags.DEFINE_float(
    'l2_reg', default=0.1, help='L2 regularization coefficient.')

flags.DEFINE_integer(
    'max_length', default=1000, help='Maximum length of examples.')

flags.DEFINE_integer(
    'num_epochs', default=200, help='Num of training epochs.')

flags.DEFINE_integer(
    'checkpoint_freq',
    default=10000,
    help='Whether to restore from existing model checkpoints.')

flags.DEFINE_bool(
    'restore_checkpoints',
    default=True,
    help='Whether to restore from existing model checkpoints.')

# Model specification.
flags.DEFINE_integer('model_dim', default=128, help='Model dimension.')

flags.DEFINE_integer(
    'num_layers', default=6, help='Number of Transformer layers.')

flags.DEFINE_integer('num_heads', default=4, help='Number of attention heads.')


def train_step(optimizer, inputs, labels, learning_rate_fn, dropout_rng):
  """Single training step."""
  dropout_rng, new_dropout_rng = random.split(dropout_rng)

  def loss_fn(model):
    """Training loss."""
    with nn.stochastic(dropout_rng):
      logits = model(inputs, train=True)
    loss = xent_loss(logits, labels)

    # L2 regularization.
    l2_loss = 0.0
    for l in range(FLAGS.num_layers):
      weights = jax.tree_leaves(model.params['Transformer1DBlock_%s' % (l + 1)])
      l2_loss += jnp.sum([jnp.sum(w ** 2) for w in weights])
    loss += l2_loss * FLAGS.l2_reg

    return loss, logits

  step = optimizer.state.step
  lr = learning_rate_fn(step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss_val, logits), grad = grad_fn(optimizer.target)
  grad = jax.lax.pmean(grad, 'batch')
  new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)
  metrics = [accuracy(logits, labels), loss_val, logits, labels]
  return new_optimizer, metrics, new_dropout_rng


def eval_step(model, inputs, labels):
  logits = model(inputs, train=False)
  return accuracy(logits, labels)


def sigmoid_xent_with_logits(logits, labels):
  return jnp.maximum(logits, 0) - logits * labels + \
        jnp.log(1 + jnp.exp(-jnp.abs(logits)))


def xent_loss(logits, labels):
  """Cross-entropy loss function."""
  losses = -jnp.sum(labels * nn.log_softmax(logits), axis=-1)
  # Average over batch.
  return jnp.mean(losses)


def accuracy(logits, labels):
  predictions = jnp.argmax(logits, axis=1)
  label_indices = jnp.argmax(labels, axis=1)
  # Num of correct samples in a batch.
  return jnp.sum(label_indices == predictions)


def main(_):
  """Builds and trains a Transformer for sentiment analysis."""

  seed = 1234
  learning_rate = FLAGS.learning_rate
  num_train_steps = FLAGS.num_train_steps
  weight_decay = FLAGS.weight_decay
  batch_size = FLAGS.batch_size
  emb_size = FLAGS.model_dim
  max_len = FLAGS.max_length
  input_shape = (batch_size, max_len)

  # Data pipeline.
  encoder, _, train_dset, test_dset = data.imdb(max_len, batch_size)

  # Model specs.
  transformer_lm_kwargs = {
    'vocab_size': encoder.vocab_size,
    'emb_dim': emb_size,
    'num_heads': FLAGS.num_heads,
    'num_layers': FLAGS.num_layers,
    'qkv_dim': FLAGS.model_dim,
    'mlp_dim': FLAGS.model_dim * 4,
    'max_len': max_len,
    'dropout_rate': FLAGS.dropout_rate,
    # Binary sentiment analysis task.
    'output_classes': 2
  }

  # Random number generator.
  rng = random.PRNGKey(seed)
  rng = jax.random.fold_in(rng, jax.host_id())
  rng, init_rng = random.split(rng)
  dropout_rngs = random.split(rng, jax.local_device_count())

  # Build Transformer.
  model, _ = build_model.create_model(init_rng, input_shape,
                                      transformer_lm_kwargs)
  optimizer = build_model.create_optimizer(model, learning_rate, weight_decay)
  del model
  start_step = 0

  # Restore ckpts.
  if FLAGS.restore_checkpoints:
    optimizer = checkpoints.restore_checkpoint(FLAGS.model_dir, optimizer)
    # Retrieve initial step.
    start_step = int(optimizer.state.step)

  # Replicate optimizer to multiple devices.
  optimizer = jax_utils.replicate(optimizer)
  # Learning rate schedule.
  learning_rate_fn = build_model.create_learning_rate_scheduler(
      base_learning_rate=learning_rate)

  # pmap training steps.
  p_train_step = jax.pmap(
      functools.partial(train_step, learning_rate_fn=learning_rate_fn),
      axis_name='batch')
  p_eval_step = jax.pmap(eval_step, axis_name='batch')

  step = start_step
  for epoch in range(FLAGS.num_epochs):
    train_generator = tfds.as_numpy(train_dset)
    logging.info('Epoch: %s' % epoch)

    for batch in train_generator:
      if batch['inputs'].shape != input_shape:
        # Ignore batches that have a different shape (usually last batch).
        continue
      inputs = common_utils.shard(batch['inputs'])
      labels = common_utils.shard(batch['labels'])
      # Convert to one-hot.
      labels = common_utils.onehot(labels, 2)
      optimizer, metrics, dropout_rngs = p_train_step(
          optimizer, inputs, labels, dropout_rng=dropout_rngs)

      # Checkpointing.
      if ((step % FLAGS.checkpoint_freq == 0 and step > 0) or
          step == num_train_steps - 1):
        if jax.host_id() == 0:
          # Save unreplicated optimizer + model state.
          checkpoints.save_checkpoint(FLAGS.model_dir,
                                      jax_utils.unreplicate(optimizer), step)

      # Evaluation.
      if step % FLAGS.eval_freq == 0 and step > 0:
        logging.info('Step %s: train accuracy=%s loss=%s',
                     step, np.sum(metrics[0]) / batch_size, np.mean(metrics[1]))
        test_generator = tfds.as_numpy(test_dset)
        all_accuracy = 0
        num_batches = 0
        for batch_test in test_generator:
          if batch_test['inputs'].shape != input_shape:
            continue
          test_inputs = common_utils.shard(batch_test['inputs'])
          test_labels = common_utils.shard(batch_test['labels'])
          # Convert to one-hot.
          test_labels = common_utils.onehot(test_labels, 2)
          test_metrics = p_eval_step(optimizer.target, test_inputs, test_labels)
          # Accumulate correct predictions across batch.
          all_accuracy = jnp.add(jnp.sum(test_metrics), all_accuracy)
          num_batches += 1
        test_acc = all_accuracy / (num_batches * batch_size)
        logging.info('Step %s: test accuracy=%s', step, test_acc)

      step += 1


if __name__ == '__main__':
  app.run(main)
