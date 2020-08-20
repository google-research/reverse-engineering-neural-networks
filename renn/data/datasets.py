"""Datasets."""

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

from renn import utils
from renn.data.tokenizers import load_tokenizer, SEP

__all__ = ['ag_news', 'goemotions', 'imdb', 'snli', 'tokenize_fun']


def pipeline(dset, preprocess_fun=utils.identity, bufsize=1024, filter_fn=None):
  """Common (standard) dataset pipeline.
  Preprocesses the data, filters it (if a filter function is specified),
  caches it, and shuffles it.

  Note: Does not batch"""

  # Apply custom preprocessing.
  dset = dset.map(preprocess_fun)

  # Apply custom filter.
  if filter_fn is not None:
    dset = dset.filter(filter_fn)

  # Cache and shuffle.
  dset = dset.cache().shuffle(buffer_size=bufsize)

  return dset


def tokenize_fun(tokenizer):
  """Standard text processing function."""
  wsp = text.WhitespaceTokenizer()
  return utils.compose(tokenizer.tokenize, wsp.tokenize, text.case_fold_utf8)


def padded_batch(dset, batch_size, sequence_length, label_shape=()):
  """Pads examples to a fixed length, and collects them into batches."""

  # We assume the dataset contains inputs, labels, and an index.
  padded_shapes = {
      'inputs': (sequence_length,),
      'labels': label_shape,
      'index': (),
  }

  # Filter out examples longer than sequence length.
  dset = dset.filter(lambda d: d['index'] <= sequence_length)

  # Pad remaining examples to the sequence length.
  dset = dset.padded_batch(batch_size, padded_shapes)

  return dset


def load_text_classification(name,
                             split,
                             preprocess_fun,
                             filter_fn=None,
                             data_dir=None):
  """Helper that loads a text classification dataset."""

  # Load raw dataset.
  dset = tfds.load(name, split=split, data_dir=data_dir)

  # Apply common dataset pipeline.
  dset = pipeline(dset, preprocess_fun=preprocess_fun, filter_fn=filter_fn)

  return dset


def ag_news(split,
            vocab_file,
            sequence_length=100,
            batch_size=64,
            transform_fn=utils.identity,
            filter_fn=None,
            data_dir=None):
  """Loads the ag news dataset."""
  tokenize = tokenize_fun(load_tokenizer(vocab_file))

  def _preprocess(d):
    """Applies tokenization."""
    tokens = tokenize(d['description']).flat_values  # Note: we ignore 'title'
    preprocessed = {
        'inputs': tokens,
        'labels': d['label'],
        'index': tf.size(tokens),
    }
    return transform_fn(preprocessed)

  # Load dataset.
  dset = load_text_classification('ag_news_subset',
                                  split,
                                  _preprocess,
                                  filter_fn,
                                  data_dir=data_dir)

  # Pad remaining examples to the sequence length.
  dset = padded_batch(dset, batch_size, sequence_length)

  return dset


def goemotions(split,
               vocab_file,
               sequence_length=50,
               batch_size=64,
               transform=utils.identity,
               filter_fn=None,
               data_dir=None):
  """Loads the goemotions dataset."""
  tokenize = tokenize_fun(load_tokenizer(vocab_file))

  emotions = ('admiration', 'amusement', 'anger', 'annoyance', 'approval',
              'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
              'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
              'gratitude', 'grief', 'joy', 'love', 'nervousness', 'neutral',
              'optimism', 'pride', 'realization', 'relief', 'remorse',
              'sadness', 'surprise')

  def _preprocess(d):
    tokens = tokenize(d['comment_text']).flat_values
    index = tf.size(tokens)
    labels = tf.convert_to_tensor([d[e] for e in emotions], dtype=tf.int64)
    preprocessed = {
        'inputs': tokens,
        'labels': labels,
        'index': index,
    }
    return transform(preprocessed)

  # Load dataset.
  dset = load_text_classification('goemotions',
                                  split,
                                  _preprocess,
                                  filter_fn,
                                  data_dir=data_dir)

  # Pad remaining examples to the sequence length.
  dset = padded_batch(dset,
                      batch_size,
                      sequence_length,
                      label_shape=(len(emotions),))

  return dset


def imdb(split,
         vocab_file,
         sequence_length=1000,
         batch_size=64,
         transform=utils.identity,
         filter_fn=None,
         data_dir=None):
  """Loads the imdb reviews dataset."""
  tokenize = tokenize_fun(load_tokenizer(vocab_file))

  def _preprocess(d):
    """Applies tokenization."""
    tokens = tokenize(d['text']).flat_values
    preprocessed = {
        'inputs': tokens,
        'labels': d['label'],
        'index': tf.size(tokens),
    }
    return transform(preprocessed)

  # Load dataset.
  dset = load_text_classification('imdb_reviews',
                                  split,
                                  _preprocess,
                                  filter_fn,
                                  data_dir=data_dir)

  # Pad remaining examples to the sequence length.
  dset = padded_batch(dset, batch_size, sequence_length)

  return dset


def snli(split,
         vocab_file,
         sequence_length=75,
         batch_size=64,
         transform=utils.identity,
         filter_fn=None,
         data_dir=None):
  """Loads the SNLI dataset."""
  tokenize = tokenize_fun(load_tokenizer(vocab_file))

  def _preprocess(d):
    """Applies tokenization."""
    hypothesis = tokenize(d['hypothesis']).flat_values
    premise = tokenize(d['premise']).flat_values
    sep = tokenize(SEP).flat_values
    tokens = tf.concat([hypothesis, sep, premise], axis=0)
    return transform({
        'inputs': tokens,
        'labels': d['label'],
        'index': tf.size(tokens),
    })

  # Load dataset.
  dset = load_text_classification('snli',
                                  split,
                                  _preprocess,
                                  filter_fn,
                                  data_dir=data_dir)

  # Pad remaining examples to the sequence length.
  dset = padded_batch(dset, batch_size, sequence_length)

  return dset


def mnist(split,
          order='row',
          batch_size=64,
          transform=utils.identity,
          filter_fn=None,
          data_dir=None):
  """Loads the MNIST data."""

  def _preprocess(example):
    """Relabels the items in an example to
    'inputs', 'labels', and 'index',
    and transposes each image if col-order is specified"""

    # Remove singleton dimension from image which is initially [28,28,1]
    image = tf.squeeze(example['image'])

    # Cast to float and make values lie within [0,1]
    image = tf.cast(image, tf.float32) / 255.

    # Subtract mean from
    #
