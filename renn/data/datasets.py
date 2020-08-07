"""Datasets."""

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

from renn import utils
from renn.data.tokenizers import load_tokenizer


__all__ = ['imdb', 'goemotions', 'tokenize_fun']


def pipeline(dset, preprocess_fun=utils.identity, bufsize=1024):
  """Common (standard) dataset pipeline."""

  # Apply custom preprocessing.
  dset = dset.map(preprocess_fun)

  # Cache and shuffle.
  dset = dset.cache().shuffle(buffer_size=bufsize)

  return dset


def tokenize_fun(tokenizer):
  """Standard text processing function."""
  wsp = text.WhitespaceTokenizer()
  return utils.compose(tokenizer.tokenize, wsp.tokenize, text.case_fold_utf8)


def goemotions(split, vocab_file, sequence_length=50, batch_size=64):
  """Loads the goemotions dataset."""
  tokenize = tokenize_fun(load_tokenizer(vocab_file))
  dset = tfds.load('goemotions', split=split)

  emotions = ('admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'neutral', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise')

  def _preprocess(d):
    tokens = tokenize(d['comment_text']).flat_values
    index = tf.size(tokens)
    labels = tf.convert_to_tensor([d[e] for e in emotions], dtype=tf.int64)
    return {
        'inputs': tokens,
        'labels': labels,
        'index': index,
    }

  # Shapes for the padded batch.
  padded_shapes = {
      'inputs': (sequence_length,),
      'labels': (len(emotions),),
      'index': (),
  }

  # Apply dataset pipeline.
  dset = pipeline(dset, preprocess_fun=_preprocess)

  # Filter out examples longer than the sequence length.
  dset = dset.filter(lambda d: d['index'] <= sequence_length)

  # Pad remaining examples to the sequence length.
  dset = dset.padded_batch(batch_size, padded_shapes)

  return dset


def imdb(split, vocab_file, sequence_length=1000, batch_size=64):
  """Loads the imdb reviews dataset."""
  tokenize = tokenize_fun(load_tokenizer(vocab_file))
  dset = tfds.load('imdb_reviews', split=split)

  def _preprocess(d):
    """Applies tokenization."""
    tokens = tokenize(d['text']).flat_values
    return {
        'inputs': tokens,
        'labels': d['label'],
        'index': tf.size(tokens),
    }

  # Shapes for the padded batch.
  padded_shapes = {
      'inputs': (sequence_length,),
      'labels': (),
      'index': (),
  }

  # Apply dataset pipeline.
  dset = pipeline(dset, preprocess_fun=_preprocess)

  # Filter out examples longer than the sequence length.
  dset = dset.filter(lambda d: d['index'] <= sequence_length)

  # Pad remaining examples to the sequence length.
  dset = dset.padded_batch(batch_size, padded_shapes)

  return dset
