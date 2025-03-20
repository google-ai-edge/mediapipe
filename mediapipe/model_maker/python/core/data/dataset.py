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
"""Common dataset for model training and evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from typing import Any, Callable, Optional, Tuple, TypeVar

import tensorflow as tf

_DatasetT = TypeVar('_DatasetT', bound='Dataset')


class Dataset(object):
  """A generic dataset class for loading model training and evaluation dataset.

  For each ML task, such as image classification, text classification etc., a
  subclass can be derived from this class to provide task-specific data loading
  utilities.
  """

  def __init__(self, tf_dataset: tf.data.Dataset, size: Optional[int] = None):
    """Initializes Dataset class.

    To build dataset from raw data, consider using the task specific utilities,
    e.g. from_folder().

    Args:
      tf_dataset: A tf.data.Dataset object that contains a potentially large set
        of elements, where each element is a pair of (input_data, target). The
        `input_data` means the raw input data, like an image, a text etc., while
        the `target` means the ground truth of the raw input data, e.g. the
        classification label of the image etc.
      size: The size of the dataset. tf.data.Dataset donesn't support a function
        to get the length directly since it's lazy-loaded and may be infinite.
    """
    self._dataset = tf_dataset
    self._size = size

  @property
  def size(self) -> Optional[int]:
    """Returns the size of the dataset.

    Same functionality as calling __len__. See the __len__ method definition for
    more information.

    Raises:
      TypeError if self._size is not set and the cardinality of self._dataset
        is INFINITE_CARDINALITY or UNKNOWN_CARDINALITY.
    """
    return self.__len__()

  def gen_tf_dataset(
      self,
      batch_size: int = 1,
      is_training: bool = False,
      shuffle: bool = False,
      preprocess: Optional[Callable[..., Any]] = None,
      drop_remainder: bool = False,
      num_parallel_preprocess_calls: int = tf.data.experimental.AUTOTUNE,
  ) -> tf.data.Dataset:
    """Generates a batched tf.data.Dataset for training/evaluation.

    Args:
      batch_size: An integer, the returned dataset will be batched by this size.
      is_training: A boolean, when True, the returned dataset will be optionally
        shuffled and repeated as an endless dataset.
      shuffle: A boolean, when True, the returned dataset will be shuffled to
        create randomness during model training.
      preprocess: A function taking three arguments in order, feature, label and
        boolean is_training.
      drop_remainder: boolean, whether the finally batch drops remainder.
      num_parallel_preprocess_calls: The number of parallel calls for dataset
        map of preprocess function.

    Returns:
      A TF dataset ready to be consumed by Keras model.
    """
    dataset = self._dataset

    if preprocess:
      preprocess = functools.partial(preprocess, is_training=is_training)
      dataset = dataset.map(
          preprocess,
          num_parallel_calls=num_parallel_preprocess_calls,
          deterministic=False,
      )

    if is_training:
      if shuffle:
        # Shuffle size should be bigger than the batch_size. Otherwise it's only
        # shuffling within the batch, which equals to not having shuffle.
        buffer_size = 3 * batch_size
        # But since we are doing shuffle before repeat, it doesn't make sense to
        # shuffle more than total available entries.
        # TODO: Investigate if shuffling before / after repeat
        # dataset can get a better performance?
        # Shuffle after repeat will give a more randomized dataset and mix the
        # epoch boundary: https://www.tensorflow.org/guide/data
        if self._size:
          buffer_size = min(self._size, buffer_size)
        dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    # TODO: Consider converting dataset to distributed dataset
    # here.
    return dataset

  def __len__(self) -> int:
    """Returns the number of element of the dataset.

    If size is not set, this method will fallback to using the __len__ method
    of the tf.data.Dataset in self._dataset. Calling __len__ on a
    tf.data.Dataset instance may throw a TypeError because the dataset may
    be lazy-loaded with an unknown size or have infinite size.

    In most cases, however, when an instance of this class is created by helper
    functions like 'from_folder', the size of the dataset will be preprocessed,
    and the _size instance variable will be already set.

    Raises:
      TypeError if self._size is not set and the cardinality of self._dataset
        is INFINITE_CARDINALITY or UNKNOWN_CARDINALITY.
    """
    if self._size is not None:
      return self._size
    else:
      return len(self._dataset)

  def split(self: _DatasetT, fraction: float) -> Tuple[_DatasetT, _DatasetT]:
    """Splits dataset into two sub-datasets with the given fraction.

    Primarily used for splitting the data set into training and testing sets.

    Args:
      fraction: A float value defines the fraction of the first returned
        subdataset in the original data.

    Returns:
      The splitted two sub datasets.
    """
    return self._split(fraction)

  def _split(
      self: _DatasetT, fraction: float, *args
  ) -> Tuple[_DatasetT, _DatasetT]:
    """Implementation for `split` method and returns sub-class instances.

    Child DataLoader classes, if requires additional constructor arguments,
    should implement their own `split` method by calling `_split` with all
    arguments to the constructor.

    Args:
      fraction: A float value defines the fraction of the first returned
        subdataset in the original data.
      *args: additional arguments passed to the sub-class constructor.

    Returns:
      The splitted two sub datasets.

    Raises:
      ValueError: if the provided fraction is not between 0 and 1.
      ValueError: if this dataset does not have a set size.
    """
    if not (fraction > 0 and fraction < 1):
      raise ValueError(f'Fraction must be between 0 and 1. Got:{fraction}')
    if not self._size:
      raise ValueError(
          'Dataset size unknown. Cannot split the dataset when '
          'the size is unknown.'
      )

    dataset = self._dataset

    train_size = int(self._size * fraction)
    trainset = self.__class__(dataset.take(train_size), *args, size=train_size)

    test_size = self._size - train_size
    testset = self.__class__(dataset.skip(train_size), *args, size=test_size)

    return trainset, testset
