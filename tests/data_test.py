# Copyright 2020 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for datasets."""

import io
import os
import pytest
import tempfile
from renn.data import datasets
from renn.data.tokenizers import load_tokenizer, punctuation_separator


@pytest.fixture
def vocab():
  tokens = ['<unk>', 'the', 'i', 'a', 'test', 'thi', '##s', '##.']

  # Write vocab to a temporary file.
  vocab_file = tempfile.NamedTemporaryFile(mode='w')
  vocab_file.file.write('\n'.join(tokens))
  vocab_file.file.flush()

  return vocab_file


def test_tokenizer_fun(vocab):
  """Tests the subword tokenizer."""
  tokenizer = load_tokenizer(vocab.name)
  tokenize = datasets.tokenize_fun(tokenizer)

  actual = list(tokenize("this is a test.").flat_values.numpy())
  expected = [5, 6, 2, 6, 3, 4, 7]
  assert actual == expected


@pytest.fixture
def sentences():
  """Test sentences for punctuation separation"""

  with_punctuation = [
      'Hello, how are you?', 'Pictures of Quebec: roofs of Quebec',
      'In a not-so-shocking survey, they discovered.',
      'linens, hair dryer, shower cabin, washing machine per arrangement',
      'Los Feliz (Greater L.A.), which nowadays belongs to actress.',
      'word1.word2', 'word1,word2', 'word1:word2', 'word1;word2', 'word1?word2',
      'word1!word2', 'word1. word2', 'word1, word2', 'word1: word2',
      'word1; word2', 'word1? word2', 'word1! word2', 'word.', 'word,', 'word:',
      'word;', 'word?', 'word!'
  ]

  processed_ideal = [
      'Hello , how are you ?', 'Pictures of Quebec : roofs of Quebec',
      'In a not-so-shocking survey , they discovered .',
      'linens , hair dryer , shower cabin , washing machine per arrangement',
      'Los Feliz (Greater L.A.) , which nowadays belongs to actress .',
      'word1.word2', 'word1,word2', 'word1:word2', 'word1;word2', 'word1?word2',
      'word1!word2', 'word1 . word2', 'word1 , word2', 'word1 : word2',
      'word1 ; word2', 'word1 ? word2', 'word1 ! word2', 'word .', 'word ,',
      'word :', 'word ;', 'word ?', 'word !'
  ]

  return (with_punctuation, processed_ideal)


def test_punctuation_separator(sentences):
  for sentence, processed in zip(*sentences):
    assert punctuation_separator(sentence) == processed
