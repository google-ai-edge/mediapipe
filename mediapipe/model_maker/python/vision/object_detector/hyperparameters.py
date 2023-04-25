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
from typing import List

from mediapipe.model_maker.python.core import hyperparameters as hp


@dataclasses.dataclass
class HParams(hp.BaseHParams):
  """The hyperparameters for training object detectors.

  Attributes:
    learning_rate: Learning rate to use for gradient descent training.
    batch_size: Batch size for training.
    epochs: Number of training iterations over the dataset.
    do_fine_tuning: If true, the base module is trained together with the
      classification layer on top.
    learning_rate_epoch_boundaries: List of epoch boundaries where
      learning_rate_epoch_boundaries[i] is the epoch where the learning rate
      will decay to learning_rate * learning_rate_decay_multipliers[i].
    learning_rate_decay_multipliers: List of learning rate multipliers which
      calculates the learning rate at the ith boundary as learning_rate *
      learning_rate_decay_multipliers[i].
  """

  # Parameters from BaseHParams class.
  learning_rate: float = 0.003
  batch_size: int = 32
  epochs: int = 10

  # Parameters for learning rate decay
  learning_rate_epoch_boundaries: List[int] = dataclasses.field(
      default_factory=lambda: []
  )
  learning_rate_decay_multipliers: List[float] = dataclasses.field(
      default_factory=lambda: []
  )

  def __post_init__(self):
    # Validate stepwise learning rate parameters
    lr_boundary_len = len(self.learning_rate_epoch_boundaries)
    lr_decay_multipliers_len = len(self.learning_rate_decay_multipliers)
    if lr_boundary_len != lr_decay_multipliers_len:
      raise ValueError(
          "Length of learning_rate_epoch_boundaries and ",
          "learning_rate_decay_multipliers do not match: ",
          f"{lr_boundary_len}!={lr_decay_multipliers_len}",
      )
    # Validate learning_rate_epoch_boundaries
    if (
        sorted(self.learning_rate_epoch_boundaries)
        != self.learning_rate_epoch_boundaries
    ):
      raise ValueError(
          "learning_rate_epoch_boundaries is not in ascending order: ",
          self.learning_rate_epoch_boundaries,
      )
    if (
        self.learning_rate_epoch_boundaries
        and self.learning_rate_epoch_boundaries[-1] > self.epochs
    ):
      raise ValueError(
          "Values in learning_rate_epoch_boundaries cannot be greater ",
          "than epochs",
      )


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

  learning_rate: float = 0.03
  batch_size: int = 32
  epochs: int = 10
  decay_steps: int = 231
  decay_rate: float = 0.96
