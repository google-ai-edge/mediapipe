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
"""Hyperparameters for training models. Shared across tasks."""

import dataclasses
import tempfile

from typing import Optional


@dataclasses.dataclass
class BaseHParams:
  """Hyperparameters used for training models.

  A common set of hyperparameters shared by the training jobs of all model
  maker tasks.

  Attributes:
    learning_rate: The learning rate to use for gradient descent training.
    batch_size: Batch size for training.
    epochs: Number of training iterations over the dataset.
    steps_per_epoch: An optional integer indicate the number of training steps
      per epoch. If not set, the training pipeline calculates the default steps
      per epoch as the training dataset size devided by batch size.
    shuffle: True if the dataset is shuffled before training.
    export_dir: The location of the model checkpoint files.
    distribution_strategy: A string specifying which Distribution Strategy to
      use. Accepted values are 'off', 'one_device', 'mirrored',
      'parameter_server', 'multi_worker_mirrored', and 'tpu' -- case
      insensitive. 'off' means not to use Distribution Strategy; 'tpu' means to
      use TPUStrategy using `tpu_address`. See the tf.distribute.Strategy
      documentation for more details:
      https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy.
    num_gpus: How many GPUs to use at each worker with the
      DistributionStrategies API. The default is -1, which means utilize all
      available GPUs.
    tpu: The Cloud TPU to use for training. This should be either the name used
      when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.
  """

  # Parameters for train configuration
  learning_rate: float
  batch_size: int
  epochs: int
  steps_per_epoch: Optional[int] = None

  # Dataset-related parameters
  shuffle: bool = False

  # Parameters for model / checkpoint files
  export_dir: str = tempfile.mkdtemp()

  # Parameters for hardware acceleration
  distribution_strategy: str = 'off'
  num_gpus: int = -1  # default value of -1 means use all available GPUs
  tpu: str = ''
