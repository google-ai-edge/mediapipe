# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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
"""Hyperparameters for training image classification models."""

import dataclasses
import tempfile
from typing import Optional


# TODO: Expose other hyperparameters, e.g. data augmentation
# hyperparameters if requested.
@dataclasses.dataclass
class HParams:
  """The hyperparameters for training image classifiers.

  The hyperparameters include:
    # Parameters about training data.
    do_fine_tuning: If true, the base module is trained together with the
      classification layer on top.
    shuffle: A boolean controlling if shuffle the dataset. Default to false.

    # Parameters about training configuration
    train_epochs: Training will do this many iterations over the dataset.
    batch_size: Each training step samples a batch of this many images.
    learning_rate: The learning rate to use for gradient descent training.
    dropout_rate: The fraction of the input units to drop, used in dropout
      layer.
    l1_regularizer: A regularizer that applies a L1 regularization penalty.
    l2_regularizer: A regularizer that applies a L2 regularization penalty.
    label_smoothing: Amount of label smoothing to apply. See tf.keras.losses for
      more details.
    do_data_augmentation: A boolean controlling whether the training dataset is
      augmented by randomly distorting input images, including random cropping,
      flipping, etc. See utils.image_preprocessing documentation for details.
    steps_per_epoch: An optional integer indicate the number of training steps
      per epoch. If not set, the training pipeline calculates the default steps
      per epoch as the training dataset size devided by batch size.
    decay_samples: Number of training samples used to calculate the decay steps
      and create the training optimizer.
    warmup_steps: Number of warmup steps for a linear increasing warmup schedule
       on learning rate. Used to set up warmup schedule by model_util.WarmUp.

    # Parameters about the saved checkpoint
    model_dir: The location of model checkpoint files and exported model files.
  """
  # Parameters about training data
  do_fine_tuning: bool = False
  shuffle: bool = False
  # Parameters about training configuration
  train_epochs: int = 5
  batch_size: int = 32
  learning_rate: float = 0.005
  dropout_rate: float = 0.2
  l1_regularizer: float = 0.0
  l2_regularizer: float = 0.0001
  label_smoothing: float = 0.1
  do_data_augmentation: bool = True
  steps_per_epoch: Optional[int] = None
  decay_samples: int = 10000 * 256
  warmup_epochs: int = 2

  # Parameters about the saved checkpoint
  model_dir: str = tempfile.mkdtemp()
