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
from typing import Sequence, Union

from mediapipe.model_maker.python.core import hyperparameters as hp
from mediapipe.model_maker.python.text.text_classifier import bert_tokenizer


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
    end_learning_rate: End learning rate for linear decay. Defaults to 0.
    batch_size: Batch size for training. Defaults to 48.
    epochs: Number of training iterations over the dataset. Defaults to 2.
    optimizer: Optimizer to use for training. Supported values are defined in
      BertOptimizer enum: ADAMW and LAMB.
    weight_decay: Weight decay of the optimizer. Defaults to 0.01.
    desired_precisions: If specified, adds a RecallAtPrecision metric per
      desired_precisions[i] entry which tracks the recall given the constraint
      on precision. Only supported for binary classification.
    desired_recalls: If specified, adds a PrecisionAtRecall metric per
      desired_recalls[i] entry which tracks the precision given the constraint
      on recall. Only supported for binary classification.
    gamma: Gamma parameter for focal loss. To use cross entropy loss, set this
      value to 0. Defaults to 2.0.
    tokenizer: Tokenizer to use for preprocessing. Must be one of the enum
      options of SupportedBertTokenizers. Defaults to FULL_TOKENIZER.
    checkpoint_frequency: Frequency(in epochs) of saving checkpoints during
      training. Defaults to 0 which does not save training checkpoints.
  """

  learning_rate: float = 3e-5
  end_learning_rate: float = 0.0

  batch_size: int = 48
  epochs: int = 2
  optimizer: BertOptimizer = BertOptimizer.ADAMW
  weight_decay: float = 0.01

  desired_precisions: Sequence[float] = dataclasses.field(default_factory=list)
  desired_recalls: Sequence[float] = dataclasses.field(default_factory=list)

  gamma: float = 2.0

  tokenizer: bert_tokenizer.SupportedBertTokenizers = (
      bert_tokenizer.SupportedBertTokenizers.FULL_TOKENIZER
  )

  checkpoint_frequency: int = 0


HParams = Union[BertHParams, AverageWordEmbeddingHParams]
