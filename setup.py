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

import os
from setuptools import setup, find_packages

global __version__
__version__ = None

with open('renn/version.py') as f:
  exec(f.read(), globals())

here = os.path.abspath(os.path.dirname(__file__))
try:
  README = open(os.path.join(here, 'README.md')).read()
except IOError:
  README = ''

setup(
    name='renn',
    version=__version__,
    description='Research tools for Reverse Engineering Neural Networks (RENN).',
    long_description=README,
    author='Niru Maheswaranathan',
    author_email='nirum@google.com',
    packages=find_packages(exclude=["examples"]),
    python_requires='>=3.7',
    install_requires=[
        'numpy >=1.12',
        'jax',
        'jaxlib',
        'msgpack',
        'sklearn',
        'tensorflow',
        'tensorflow-text',
        'tfds-nightly',
        'tqdm',
    ],
    url='https://github.com/google-research/reverse-engineering-neural-networks',
    license='Apache-2.0',
)
