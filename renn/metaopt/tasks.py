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

"""Load tasks from the library."""

from .task_lib import classification
from .task_lib import quadratic
from .task_lib import regression

__all__ = [
    'fc',
    'cnn',
    'linreg',
    'logreg',
    'clf',
    'quad'
]

# Pre-defined neural network models.
fc = classification.fully_connected
cnn = classification.conv2d

# Each task below is a function that takes problem hyperparameters as
# arguments and returns a `problem_fun` function. This function takes a
# single argument, a PRNGKey, and returns a (x_init, loss_fun, data) tuple.
# x_init is a pytree initial problem parameters. loss_fun is a function
# that returns a scalar loss given parameters and a batch of data. Finally,
# data is an iterable pytree. All leaves of the tree must have the same first
# dimension, which is the number of steps to optimize for. These slices
# of data will be passed to the loss_fun during optimization.
linreg = regression.linear
logreg = regression.logistic
clf = classification.classifier
quad = quadratic.loguniform
