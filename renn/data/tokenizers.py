"""Text processing."""

from collections import Counter
import itertools

from renn.data import wordpiece_tokenizer_learner_lib as vocab_learner

import tensorflow_text as text
import tensorflow as tf
import tensorflow.strings as strings

import re

__all__ = ['build_vocab', 'load_tokenizer']

# Special tokens
JOINER = '##'
UNK = '<unk>'
CLS = '<cls>'
SEP = '<sep>'
EOS = '<eos>'
BOS = '<bos>'


def _get_transform_fn(lower: bool, separate_punctuation: bool):
  """Helper function for text_generator"""

  if not lower and not separate_punctuation:
    return lambda x: x
  elif lower and not separate_punctuation:
    return lambda x: x.lower()
  elif not lower and separate_punctuation:
    return lambda x: _punctuation_separator(x)
  elif lower and separate_punctuation:
    return lambda x: _punctuation_separator(x.lower())


def _punctuation_separator(s: str) -> str:
  """Separates punctuation at the end of word and end of line"""
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


def _tensor_punctuation_separator(x: tf.Tensor) -> tf.Tensor:
  """Separates punctuation at the end of word and end of line.
  In behavior this function is identical to _punctuation_separator
  above.  The only difference is that this acts on TF Tensors rather
  than strings. """
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


def text_generator(dataset,
                   split,
                   language,
                   num_examples,
                   lower=True,
                   separate_punctuation=True):
  """Given a dataset (formatted for TFDS paracrawl translation dataset,
  returns a generator which yields single-language examples from that
  dataset, one at a time as strings

  Arguments:
    dataset - TFDS dataset
    split - e.g., 'train', 'test'.  dataset[split] should yield an iterable
    language - which language to generate text from.
               dataset[split] items should have the language as a key
    num_examples - desired length of the generator
    lower - bool, whether to lowercase each sentence
  """

  transform_fn = _get_transform_fn(lower, separate_punctuation)

  it = iter(dataset[split])
  for count in range(num_examples):
    yield transform_fn(next(it)[language].numpy().decode('UTF-8'))


def build_vocab_tr(corpus_generator, vocab_size, split_fun=str.split):
  """Builds a vocab file from a text generator for translation.
  Unlike buld_vocab() below, these vocabularies will have 3
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
