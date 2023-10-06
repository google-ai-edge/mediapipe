# Copyright 2022 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hyperparameters for training customized gesture recognizer models."""

import dataclasses

from mediapipe.model_maker.python.core import hyperparameters as hp


@dataclasses.dataclass
class HParams(hp.BaseHParams):
  """The hyperparameters for training gesture recognizer.

  Attributes:
    learning_rate: Learning rate to use for gradient descent training.
    batch_size: Batch size for training.
    epochs: Number of training iterations over the dataset.
    lr_decay: Learning rate decay to use for gradient descent training.
    gamma: Gamma parameter for focal loss.
  """
  # Parameters from BaseHParams class.
  learning_rate: float = 0.001
  batch_size: int = 2
  epochs: int = 10

  # Parameters about training configuration
  # TODO: Move lr_decay to hp.baseHParams.
  lr_decay: float = 0.99
  gamma: int = 2
