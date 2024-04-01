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
"""Hyperparameters for training image classification models."""

import dataclasses

from mediapipe.model_maker.python.core import hyperparameters as hp


@dataclasses.dataclass
class HParams(hp.BaseHParams):
  """The hyperparameters for training image classifiers.

  Attributes:
    learning_rate: Learning rate to use for gradient descent training.
    batch_size: Batch size for training.
    epochs: Number of training iterations over the dataset.
    do_fine_tuning: If true, the base module is trained together with the
      classification layer on top.
    l1_regularizer: A regularizer that applies a L1 regularization penalty.
    l2_regularizer: A regularizer that applies a L2 regularization penalty.
    label_smoothing: Amount of label smoothing to apply. See tf.keras.losses for
      more details.
    do_data_augmentation: A boolean controlling whether the training dataset is
      augmented by randomly distorting input images, including random cropping,
      flipping, etc. See utils.image_preprocessing documentation for details.
    decay_samples: Number of training samples used to calculate the decay steps
      and create the training optimizer.
    warmup_steps: Number of warmup steps for a linear increasing warmup schedule
      on learning rate. Used to set up warmup schedule by model_util.WarmUp.
    checkpoint_frequency: Frequency to save checkpoint.
    one_hot: Whether the label data is score input or one-hot.
    multi_labels: Whether the model predict multi labels.
  """
  # Parameters from BaseHParams class.
  learning_rate: float = 0.001
  batch_size: int = 2
  epochs: int = 10
  # Parameters about training configuration
  do_fine_tuning: bool = False
  l1_regularizer: float = 0.0
  l2_regularizer: float = 0.0001
  label_smoothing: float = 0.1
  do_data_augmentation: bool = True
  # TODO: Use lr_decay in hp.baseHParams to infer decay_samples.
  decay_samples: int = 10000 * 256
  warmup_epochs: int = 2
  checkpoint_frequency: int = 1
  one_hot: bool = True
  multi_labels: bool = False
