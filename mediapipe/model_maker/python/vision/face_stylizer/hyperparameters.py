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
"""Hyperparameters for training on-device face stylization models."""

import dataclasses

from mediapipe.model_maker.python.core import hyperparameters as hp


@dataclasses.dataclass
class HParams(hp.BaseHParams):
  """The hyperparameters for training face stylizers.

  Attributes:
    learning_rate: Learning rate to use for gradient descent training.
    batch_size: Batch size for training.
    epochs: Number of training iterations.
    beta_1: beta_1 used in tf.keras.optimizers.Adam.
    beta_2: beta_2 used in tf.keras.optimizers.Adam.
  """

  # Parameters from BaseHParams class.
  learning_rate: float = 8e-4
  batch_size: int = 4
  epochs: int = 100
  # Parameters for face stylizer.
  beta_1 = 0.0
  beta_2 = 0.99
