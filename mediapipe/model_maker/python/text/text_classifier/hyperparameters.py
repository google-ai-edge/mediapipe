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
import enum
from typing import Union

from mediapipe.model_maker.python.core import hyperparameters as hp


@dataclasses.dataclass
class AverageWordEmbeddingHParams(hp.BaseHParams):
  """The hyperparameters for an AverageWordEmbeddingClassifier."""


@enum.unique
class BertOptimizer(enum.Enum):
  """Supported Optimizers for Bert Text Classifier."""

  ADAMW = "adamw"
  LAMB = "lamb"


@dataclasses.dataclass
class BertHParams(hp.BaseHParams):
  """The hyperparameters for a Bert Classifier.

  Attributes:
    learning_rate: Learning rate to use for gradient descent training.
    batch_size: Batch size for training.
    epochs: Number of training iterations over the dataset.
    optimizer: Optimizer to use for training. Only supported values are "adamw"
      and "lamb".
  """

  learning_rate: float = 3e-5
  batch_size: int = 48
  epochs: int = 2
  optimizer: BertOptimizer = BertOptimizer.ADAMW


HParams = Union[BertHParams, AverageWordEmbeddingHParams]
