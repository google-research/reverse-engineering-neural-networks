"""Text processing."""

from collections import Counter
import itertools

from renn.data import wordpiece_tokenizer_learner_lib as vocab_learner

import tensorflow_text as text
import tensorflow as tf

__all__ = ['build_vocab', 'load_tokenizer']

# Special tokens
JOINER = '##'
UNK = '<unk>'
CLS = '<cls>'
SEP = '<sep>'


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
  table = tf.lookup.TextFileInitializer(vocab_file, tf.string,
                                        tf.lookup.TextFileIndex.WHOLE_LINE,
                                        tf.int64,
                                        tf.lookup.TextFileIndex.LINE_NUMBER)
  static_table = tf.lookup.StaticHashTable(table, default_value)

  # Build tokenizer.
  tokenizer = text.WordpieceTokenizer(static_table,
                                      suffix_indicator=JOINER,
                                      max_bytes_per_word=100,
                                      max_chars_per_token=None,
                                      token_out_type=tf.int64,
                                      unknown_token=UNK)

  return tokenizer
