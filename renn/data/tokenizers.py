"""Text processing."""

import tensorflow_datasets as tfds
import tensorflow_text as tft


build_vocab = tfds.features.text.SubwordTextEncoder.build_from_corpus
load_vocab = tfds.features.text.SubwordTextEncoder.load_from_file



