"""Datasets."""

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

import os

from renn import utils
from renn.data.tokenizers import load_tokenizer, SEP
from renn.data import data_utils

__all__ = [
    'ag_news', 'goemotions', 'imdb', 'snli', 'tokenize_fun', 'mnist', 'yelp',
    'dbpedia', 'amazon'
]


def pipeline(dset, preprocess_fun=utils.identity, filter_fn=None, bufsize=1024):
  """Common (standard) dataset pipeline.
  Preprocesses the data, filters it (if a filter function is specified), caches it, and shuffles it.

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


def load_tfds(name, split, preprocess_fun, filter_fn=None, data_dir=None):
  """Helper that loads a text classification dataset
  from tensorflow_datasets"""

  # Load raw dataset.
  dset = tfds.load(name, split=split, data_dir=data_dir)

  # Apply common dataset pipeline.
  dset = pipeline(dset, preprocess_fun=preprocess_fun, filter_fn=filter_fn)

  return dset


def load_csv(name, split, preprocess_fun, filter_fn=None, data_dir=None):
  """Helper that loads a text classification dataset
  from a csv file"""
  # Load raw dataset.

  output_types = {'text': tf.string, 'label': tf.int64}
  output_shapes = {'text': (), 'label': ()}

  filename = os.path.join(data_dir, f'{split}.csv')
  row_parser = data_utils.PARSERS[name]
  dset_iterator = lambda: data_utils.readfile(filename, row_parser)
  dset = tf.data.Dataset.from_generator(dset_iterator, output_types,
                                        output_shapes)

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
  dset = load_tfds('ag_news_subset',
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
               emotions=None,
               transform=utils.identity,
               filter_fn=None,
               data_dir=None):
  """Loads the goemotions dataset."""
  tokenize = tokenize_fun(load_tokenizer(vocab_file))

  if emotions is None:  # Use all emotions.
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
  dset = load_tfds('goemotions',
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
  dset = load_tfds('imdb_reviews',
                   split,
                   _preprocess,
                   filter_fn,
                   data_dir=data_dir)

  # Pad remaining examples to the sequence length.
  dset = padded_batch(dset, batch_size, sequence_length)

  return dset


def yelp(split,
         num_classes,
         vocab_file,
         sequence_length=1000,
         batch_size=64,
         transform=utils.identity,
         filter_fn=None,
         data_dir=None):
  """Loads the yelp reviews dataset."""
  tokenize = tokenize_fun(load_tokenizer(vocab_file))

  if data_dir is None:
    raise ValueError('Yelp dataset requires data_dir to be provided.')

  label_conversion = data_utils.sentiment_relabel(num_classes)

  def _preprocess(d):
    """Applies tokenization, and
    transforms the yelp labels according to the
    specified number of classes"""

    tokens = tokenize(d['text']).flat_values
    preprocessed = {
        'inputs': tokens,
        'labels': label_conversion(d['label']),
        'index': tf.size(tokens),
    }

    return transform(preprocessed)

  filter_fn = lambda x: x['labels'] != -1

  # Load dataset.
  dset = load_csv('yelp', split, _preprocess, filter_fn, data_dir=data_dir)

  # Pad remaining examples to the sequence length.
  dset = padded_batch(dset, batch_size, sequence_length)

  return dset


def amazon(split,
           num_classes,
           vocab_file,
           sequence_length=250,
           batch_size=64,
           transform=utils.identity,
           filter_fn=None,
           data_dir=None):
  """Loads the yelp reviews dataset."""
  tokenize = tokenize_fun(load_tokenizer(vocab_file))

  if data_dir is None:
    raise ValueError('Amazon dataset requires data_dir to be provided.')

  label_conversion = data_utils.sentiment_relabel(num_classes)

  def _preprocess(d):
    """Applies tokenization, and
    transforms the Amazon labels according to the
    specified number of classes"""

    tokens = tokenize(d['text']).flat_values
    preprocessed = {
        'inputs': tokens,
        'labels': label_conversion(d['label']),
        'index': tf.size(tokens),
    }

    return transform(preprocessed)

  filter_fn = lambda x: x['labels'] != -1

  # Load dataset.
  dset = load_csv('amazon', split, _preprocess, filter_fn, data_dir=data_dir)

  # Pad remaining examples to the sequence length.
  dset = padded_batch(dset, batch_size, sequence_length)

  return dset


def dbpedia(split,
            num_classes,
            vocab_file,
            sequence_length=1000,
            batch_size=64,
            transform=utils.identity,
            filter_fn=None,
            data_dir=None):
  """Loads the dpedia text classification dataset."""
  tokenize = tokenize_fun(load_tokenizer(vocab_file))

  if data_dir is None:
    raise ValueError('DBPedia dataset requires data_dir to be provided.')

  def _preprocess(d):
    """Applies tokenization, and
    transforms the dbpedia labels according to the
    specified number of classes

    For a given number of classes, the classes with
    labels below that number are kept, and all other classes
    are removed.

    So, e.g., num_classes = 4, would keep classes 0,1,2,3"""

    def relabel(label):
      if label <= num_classes:
        # in DBPedia csv file, labels are
        # given as 1, 2, ...
        return label - 1
      else:
        return tf.constant(-1, dtype=tf.int64)

    tokens = tokenize(d['text']).flat_values
    preprocessed = {
        'inputs': tokens,
        'labels': relabel(d['label']),
        'index': tf.size(tokens),
    }

    return transform(preprocessed)

  filter_fn = lambda x: x['labels'] != -1

  # Load dataset.
  dset = load_csv('dbpedia', split, _preprocess, filter_fn, data_dir=data_dir)

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
  dset = load_tfds('snli', split, _preprocess, filter_fn, data_dir=data_dir)

  # Pad remaining examples to the sequence length.
  dset = padded_batch(dset, batch_size, sequence_length)

  return dset


def mnist(split,
          order='row',
          batch_size=64,
          transform=utils.identity,
          filter_fn=None,
          data_dir=None,
          classes=None):
  """Loads the serialized MNIST dataset.

  Args:
    classes - the subset of classes to keep.
              If None, all will be kept"""

  def _preprocess(example):
    image = tf.squeeze(example['image'])
    image = tf.cast(image, tf.float32) / 255.

    if order == 'col':
      image = tf.transpose(image, perm=[1, 0])

    return transform({'inputs': image, 'labels': example['label'], 'index': 28})

  # Load dataset.
  dset = tfds.load('mnist', data_dir=data_dir)[split]

  if classes is not None:
    # Filter out images without the proper label
    allowed_fn = _in_subset(classes)
    # Remap labels to be in range (0, number of classes)
    relabel_fn = _relabel_subset(classes)

    dset = dset.filter(allowed_fn).map(relabel_fn)

  dset = pipeline(dset, _preprocess, filter_fn)

  # Batch dataset.
  return dset.batch(batch_size)


def _relabel_subset(subclasses):
  """Provides a function for relabeling classes.
  Example should contain key 'label' """

  s = tf.constant(subclasses, dtype=tf.int64)

  def relabel(example):
    example.update({'label': tf.where(s == example['label'])[0][0]})
    return example

  return relabel


def _in_subset(subclasses):
  """Provides a function for determining whether
  an example is in one of the provided subclasses.
  Expmle should contain a key 'label' """

  s = tf.constant(subclasses, dtype=tf.int64)

  def in_subset(example):
    label = example['label']
    isallowed = tf.equal(s, label)
    reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))
    return tf.greater(reduced, tf.constant(0.))

  return in_subset
