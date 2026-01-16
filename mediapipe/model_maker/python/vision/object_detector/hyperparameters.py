# Copyright 2023 The MediaPipe Authors.
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
"""Hyperparameters for training object detection models."""

import dataclasses
from typing import Optional

from mediapipe.model_maker.python.core import hyperparameters as hp


@dataclasses.dataclass
class HParams(hp.BaseHParams):
  """The hyperparameters for training object detectors.

  Attributes:
    learning_rate: Learning rate to use for gradient descent training.
    batch_size: Batch size for training.
    epochs: Number of training iterations over the dataset.
    cosine_decay_epochs: The number of epochs for cosine decay learning rate.
      See
      https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay
        for more info.
    cosine_decay_alpha: The alpha value for cosine decay learning rate. See
      https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay
        for more info.
  """

  # Parameters from BaseHParams class.
  learning_rate: float = 0.3
  batch_size: int = 8
  epochs: int = 30

  # Parameters for cosine learning rate decay
  cosine_decay_epochs: Optional[int] = None
  cosine_decay_alpha: float = 1.0


@dataclasses.dataclass
class QATHParams:
  """The hyperparameters for running quantization aware training (QAT) on object detectors.

  For more information on QAT, see:
    https://www.tensorflow.org/model_optimization/guide/quantization/training

  Attributes:
    learning_rate: Learning rate to use for gradient descent QAT.
    batch_size: Batch size for QAT.
    epochs: Number of training iterations over the dataset.
    decay_steps: Learning rate decay steps for Exponential Decay. See
      https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay
        for more information.
    decay_rate: Learning rate decay rate for Exponential Decay. See
      https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay
        for more information.
  """

  learning_rate: float = 0.3
  batch_size: int = 8
  epochs: int = 15
  decay_steps: int = 8
  decay_rate: float = 0.96
