# Copyright 2020 Google LLC
#
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
"""Serialization of pytrees."""

import enum
import numpy as np
import msgpack

from .rnn.cells import LinearRNN

__all__ = ['dump', 'load', 'dumps', 'loads']


def dump(pytree, file):
  return msgpack.pack(pytree, file, default=_ext_pack, strict_types=True)


def dumps(pytree):
  return msgpack.packb(pytree, default=_ext_pack, strict_types=True)


def load(file):
  return msgpack.unpack(file, ext_hook=_ext_unpack, raw=False)


def loads(bytes):
  return msgpack.unpackb(bytes, ext_hook=_ext_unpack, raw=False)


class _CustomExtType(enum.IntEnum):
  ndarray = 1
  tuple = 2
  linear_rnn = 3


def ndarray_to_bytes(arr):
  """Converts a numpy ndarray to bytes."""

  if arr.dtype.hasobject or arr.dtype.isalignedstruct:
    raise ValueError('Object and structured dtypes are not supported.')

  data = (arr.tobytes(), arr.dtype.str, arr.shape)
  return msgpack.packb(data, use_bin_type=True)


def bytes_to_ndarray(data):
  buffer, dtype, shape = msgpack.unpackb(data, raw=False)
  return np.frombuffer(buffer, dtype=dtype).reshape(shape)


def _ext_pack(x):
  if isinstance(x, np.ndarray):
    return msgpack.ExtType(_CustomExtType.ndarray, ndarray_to_bytes(x))

  elif isinstance(x, tuple):
    return msgpack.ExtType(_CustomExtType.tuple, dumps(list(x)))

  elif isinstance(x, LinearRNN):
    return msgpack.ExtType(_CustomExtType.linear_rnn, dumps(x.flatten()))

  return x


def _ext_unpack(code, data):
  if code == _CustomExtType.ndarray:
    return bytes_to_ndarray(data)

  elif code == _CustomExtType.tuple:
    return tuple(loads(data))

  elif code == _CustomExtType.linear_rnn:
    return LinearRNN(*loads(data))

  return msgpack.ExtType(code, data)
