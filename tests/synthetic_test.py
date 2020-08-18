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

"""Tests for synthetic datasets."""

from renn.data import synthetic

def test_constant_sampler():
  """Tests that the constant sampler returns
  an array of the specified length, with the single
  specified value"""
  constant_values = [4,23,45,1,94]
  sample_nums = [10,15,16,18]

  for val in constant_values:
    s = synthetic.constant_sampler(value=val)
    for num in sample_nums:
      assert len(s(num)) == num
      assert len(set(s(num))) == 1
      assert s(num)[0] == val

def test_uniform_sampler():
  """Tests that the uniform sampler returns
  samples of the proper length, whose values lie in the
  specified interval"""
  intervals = [(10,20), (15,30)]
  sample_nums = [10,15,16,18]

  for interval in intervals:
    s = synthetic.uniform_sampler(min_val=interval[0],
                                  max_val=interval[1])
    for num in sample_nums:
      samples = s(num)
      assert len(samples) == num

      for sample in samples:
        assert sample <= interval[1] and sample >= interval[0]

