"""Datasets."""

import tensorflow_datasets as tfds

from renn import utils
from renn.data.tokenizers import load_tokenizer


__all__ = ['imdb']


def pipeline(dset, preprocess_fun=utils.identity, bufsize=1024):
  """Common (standard) dataset pipeline."""

  # Apply custom preprocessing.
  dset = dset.map(preprocess_fun)

  # Cache and shuffle.
  dset = dset.cache().shuffle(buffer_size=bufsize)

  return dset


def imdb(split, vocab_file, sequence_length=1000, batch_size=64):
  tokenizer = load_tokenizer(vocab_file)
  dset = tfds.load('imdb_reviews', split=split)

  def _preprocess(d):
    """Applies tokenization."""
    return {
        'inputs': tokenizer.tokenize(d['text']),
        'labels': d['label'],
        'index': len(d['text']),
    }

  # Shapes for the padded batch.
  padded_shapes = {
      'inputs': (sequence_length,),
      'labels': (),
      'index': (),
  }

  # Apply dataset pipeline.
  dset = dset.filter(lambda d: len(d['text']) <= sequence_length)
  dset = pipeline(dset, preprocess_fun=_preprocess)
  dset = dset.padded_batch(batch_size, padded_shapes)

  return dset
