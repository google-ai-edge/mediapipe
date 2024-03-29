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
"""Hyperparameters for training models. Shared across tasks."""

import dataclasses
import tempfile
from typing import Mapping, Optional

import tensorflow as tf

from official.common import distribute_utils


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
      per epoch as the training dataset size divided by batch size.
    class_weights: An optional mapping of indices to weights for weighting the
      loss function during training.
    shuffle: True if the dataset is shuffled before training.
    repeat: True if the training dataset is repeated infinitely to support
      training without checking the dataset size.
    export_dir: The location of the model checkpoint files.
    distribution_strategy: A string specifying which Distribution Strategy to
      use. Accepted values are 'off', 'one_device', 'mirrored',
      'parameter_server', 'multi_worker_mirrored', and 'tpu' -- case
      insensitive. 'off' means not to use Distribution Strategy; 'tpu' means to
      use TPUStrategy using `tpu_address`. See the tf.distribute.Strategy
      documentation for more details:
      https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy.
    num_gpus: How many GPUs to use at each worker with the
      DistributionStrategies API. The default is 0.
    tpu: The TPU resource to be used for training. This should be either the
      name used when creating the Cloud TPU, a grpc://ip.address.of.tpu:8470
      url, or an empty string if using a local TPU.
  """

  # Parameters for train configuration
  learning_rate: float
  batch_size: int
  epochs: int
  steps_per_epoch: Optional[int] = None
  class_weights: Optional[Mapping[int, float]] = None

  # Dataset-related parameters
  shuffle: bool = False
  repeat: bool = False

  # Parameters for model / checkpoint files
  export_dir: str = tempfile.mkdtemp()

  # Parameters for hardware acceleration
  distribution_strategy: str = 'off'
  num_gpus: int = 0
  tpu: str = ''
  _strategy: tf.distribute.Strategy = dataclasses.field(init=False)

  def __post_init__(self):
    self._strategy = distribute_utils.get_distribution_strategy(
        distribution_strategy=self.distribution_strategy,
        num_gpus=self.num_gpus,
        tpu_address=self.tpu,
    )

  def get_strategy(self):
    return self._strategy
