"""Text processing."""

from collections import Counter
import itertools

from renn.data import wordpiece_tokenizer_learner_lib as vocab_learner

import tensorflow_text as text
import tensorflow as tf
import tensorflow.strings as strings

from typing import Optional, Callable

import re

__all__ = ['build_vocab', 'load_tokenizer']

# Special tokens
JOINER = '##'
UNK = '<unk>'
CLS = '<cls>'
SEP = '<sep>'
EOS = '<eos>'
BOS = '<bos>'

def punctuation_separator(s: str) -> str:
  """Separates punctuation at the end of word and end of line."""
  punctuation_chars = ['.', ',', ':', ';', '?', '!']
  special_chars = ['.', '?']  # regex chars
  s_ = s
  for c in punctuation_chars:
    if c in special_chars:
      # separate punctuation at end-of-line
      s_ = re.sub(f'\{c}$', f' {c}', s_)
      # separate punctuation at end-of-word
      s_ = re.sub(f'\{c} ', f' {c} ', s_)
    else:
      # separate punctuation at end-of-line
      s_ = re.sub(f'{c}$', f' {c}', s_)
      # separate punctuation at end-of-word
      s_ = re.sub(f'{c} ', f' {c} ', s_)
  return s_


def tensor_punctuation_separator(x: tf.Tensor) -> tf.Tensor:
  """Separates punctuation at the end of word and end of line.

  In behavior this function is identical to punctuation_separator
  above. The only difference is that this acts on TF Tensors rather
  than strings."""
  punctuation_chars = ['.', ',', ':', ';', '?', '!']
  special_chars = ['.', '?']  # regex chars

  for c in punctuation_chars:
    if c in special_chars:
      # separate punctuation at end-of-line
      x = strings.regex_replace(x, f'\{c}$', f' {c}')
      # separate punctuation at end-of-word
      x = strings.regex_replace(x, f'\{c} ', f' {c} ')
    else:
      # separate punctuation at end-of-line
      x = strings.regex_replace(x, f'{c}$', f' {c}')
      # separate punctuation at end-of-word
      x = strings.regex_replace(x, f'{c} ', f' {c} ')
  return x

def lowercase_strip(x: str) -> str:
  """Lowercases and strips punctuation.

  Can be used as a transform_fn for text_generator()."""

  return punctuation_separator(x.lower())

def text_generator(dataset: dict,
                   split: str,
                   language: str,
                   num_examples: int,
                   transform_fn: Optional[Callable[[str],str]]=None):
  """Builds a generator from a TF dataset.

  Given a dataset, returns a generator which yields single-language examples
  from that dataset, one at a time as strings.

  Arguments:
    dataset: dictionary of datasets.
    split: 'train', 'test', etc.  dataset[split] should yield an iterable.
    language: which language to generate text from.
    transform_fn: string transformation, defaults to identity fn.
    num_examples: desired length of the generator.
  """

  # transform_fn defaults to identity
  if transform_fn is None:
    transform_fn = lambda x: x

  it = iter(dataset[split])
  for count in range(num_examples):
    yield transform_fn(next(it)[language].numpy().decode('UTF-8'))

def build_vocab_tr(corpus_generator, vocab_size, split_fun=str.split):
  """Builds a vocab file from a text generator for translation.

  Unlike build_vocab() below, these vocabularies will have 3
  reserved tokens:
    <unk> - unknown
    <bos> - beginning of sentence
    <eos> - end of sentence
  This also does not include a joiner token.
  """

  # Split documents into words.
  words = itertools.chain(*map(split_fun, corpus_generator))

  # Count words in the corpus.
  word_counts = Counter(words)

  # Find the most common words
  most_common_words = sorted(word_counts, key=word_counts.get,
                             reverse=True)[:vocab_size]

  reserved_tokens = [UNK, EOS, BOS]
  vocab = reserved_tokens + list(most_common_words)

  return vocab


def build_vocab(corpus_generator, vocab_size, split_fun=str.split):
  """Builds a vocab file from a text generator."""

  # Split documents into words.
  words = itertools.chain(*map(split_fun, corpus_generator))

  # Count words in the corpus.
  word_counts = Counter(words)

  # Specify parameters.
  reserved_tokens = (UNK, CLS, SEP)
  params = vocab_learner.Params(upper_thresh=10000000,
                                lower_thresh=10,
                                num_iterations=4,
                                max_input_tokens=5000000,
                                max_token_length=50,
                                max_unique_chars=1000,
                                vocab_size=vocab_size,
                                slack_ratio=0.05,
                                include_joiner_token=True,
                                joiner=JOINER,
                                reserved_tokens=reserved_tokens)

  # Build the vocabulary.
  vocab = vocab_learner.learn(word_counts.items(), params)

  return vocab


def load_tokenizer(vocab_file, default_value=-1):
  """Loads a tokenizer from a vocab file."""

  # Build lookup table that maps subwords to ids.
  table = tf.lookup.TextFileInitializer(
      filename=vocab_file,
      key_dtype=tf.string,
      key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
      value_dtype=tf.int64,
      value_index=tf.lookup.TextFileIndex.LINE_NUMBER)
  static_table = tf.lookup.StaticHashTable(table, default_value)

  # Build tokenizer.
  tokenizer = text.WordpieceTokenizer(static_table,
                                      suffix_indicator=JOINER,
                                      max_bytes_per_word=100,
                                      max_chars_per_token=None,
                                      token_out_type=tf.int64,
                                      unknown_token=UNK)

  return tokenizer
