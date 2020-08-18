"""Synthetic Datasets."""

import numpy as np
from itertools import product
from renn import utils

__all__ = ['Unordered']

class Unordered:
  """Synthetic dataset representing un-ordered classes, to mimic e.g.
  text-classification datasets like AG News (unlike, say, star-prediction or
  sentiment analysis, which features ordered classes"""

  SAMPLERS = {'Constant': constant_sampler,
              'Uniform': uniform_sampler}

  def __init__(self, num_classes=3, batch_size=64, length_sampler='Constant',
               sampler_params={'value': 40}):

    self.num_classes = num_classes
    self.batch_size = batch_size

    if length_sampler in SAMPLERS.keys():
      self.sampler = SAMPLERS[length_sampler](**sampler_params)
    else:
      raise ValueError(f'length_sampler must be one of {SAMPLERS.keys()}')

  def __iter__(self):
    return self

  def __next__(self):
    """ Samples and returns a batch.  As with the real datasets,
    batch is a dictionary containing 'inputs', 'labels', and 'index' keys.
    'index' specifies the length of the sequence """
    lengths = self.sampler(num_samples=self.batch_size)
    max_length = max(lengths)

    batch = {'inputs': np.random.randint(len(self.vocab),
                                         size=(self.batch_size, max_length)),
             'index': lengths}

    batch['labels'] = self.label_batch(batch)

    return batch

  def label_batch(self, batch):
    """ Calculates class labels for a batch of sentences """
    zipped = zip(batch['inputs'], batch['index'])

    class_scores =  np.array([self.score(s, l) for s, l in zipped])
    return np.argmax(class_scores, axis=1)

  def score(self, sentence, length):
    """ Calculates the score, i.e. the amount of accumulated
    evidence in the sentence, for each class"""
    return sum([self.vocab[word] for word in sentence[:length]])

  def build_vocab(self):
    """ Builds the vocabulary
    Vocabulary for this dataset consists of tuples, e.g., ('very', 3),
      indicating in this case a token which provides strong evidence of class 3.
    The degree of evidence for each basic word is stored in the WORD_VALENCES
      dictionary
    """
    WORD_VALENCES = {'very': 2,
                   'some': 1,
                   'neutral': 0,
                   'not': -1}

    words = product(WORD_VALENCES, range(self.num_classes))

    def _valence(word):
      """Converts a word like ('very', 1) to a
      vector-valued valence, in this case (0,2,0,...)"""
      valence = np.zeros(self.num_classes)
      valence[word[1]] = WORD_VALENCES[word[0]]
      return valence

    self.vocab = {i: _valence(words) for i, word in enumerate(words)}

def constant_sampler(value):
  # returns a sampling function which always returns the value 'value'

  def sample(num_samples):
    return np.fill((num_samples, ), length)
  return sample

def uniform_sampler(min_val, max_val):
  # returns a sampling function which samples uniformly between min_val and
  # max_val, inclusive

  def sample(num_samples):
    return np.random.randint(min_val, max_val+1, size=(num_samples,))
  return sample
