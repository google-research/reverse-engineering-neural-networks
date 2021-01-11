"""Text processing."""

from collections import Counter
import itertools

from renn.data import wordpiece_tokenizer_learner_lib as vocab_learner

import tensorflow_text as text
import tensorflow as tf
import tf.lookup.TextFileIndex as TFI

__all__ = ['build_vocab', 'load_tokenizer']

# Special tokens
JOINER = '##'
UNK = '<unk>'
CLS = '<cls>'
SEP = '<sep>'
EOS = '<eos>'
BOS = '<bos>'


def text_generator(dataset, split, language, num_examples, lower=True):
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

  it = iter(dataset[split])
  if lower:
    for count in range(num_examples):
      yield next(it)[language].numpy().decode('UTF-8').lower()
  else:
    for count in range(num_examples):
      yield next(it)[language].numpy().decode('UTF-8')


def build_vocab_tr(corpus_generator, vocab_size, split_fun=str.split):
  """Builds a vocab file from a text generator for translation.
  Unlike buld_vocab() below, these vocabularies will have 3
  reserved tokens:
    <unk> - unknown
    <bos> - beginning of sentence
    <eos> - end of sentence
  This also does not include a joiner token"""

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
  table = tf.lookup.TextFileInitializer(filename=vocab_file,
                                        key_dtype=tf.string,
                                        key_index=TFI.WHOLE_LINE,
                                        value_dtype=tf.int64,
                                        value_index=TFI.LINE_NUMBER)
  static_table = tf.lookup.StaticHashTable(table, default_value)

  # Build tokenizer.
  tokenizer = text.WordpieceTokenizer(static_table,
                                      suffix_indicator=JOINER,
                                      max_bytes_per_word=100,
                                      max_chars_per_token=None,
                                      token_out_type=tf.int64,
                                      unknown_token=UNK)

  return tokenizer
