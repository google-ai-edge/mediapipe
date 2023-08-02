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
"""Text classifier dataset library."""

import csv
import dataclasses
import hashlib
import os
import random
import tempfile
from typing import List, Optional, Sequence

import tensorflow as tf

from mediapipe.model_maker.python.core.data import cache_files as cache_files_lib
from mediapipe.model_maker.python.core.data import classification_dataset


@dataclasses.dataclass
class CSVParameters:
  """Parameters used when reading a CSV file.

  Attributes:
    text_column: Column name for the input text.
    label_column: Column name for the labels.
    fieldnames: Sequence of keys for the CSV columns. If None, the first row of
      the CSV file is used as the keys.
    delimiter: Character that separates fields.
    quotechar: Character used to quote fields that contain special characters
      like the `delimiter`.
  """
  text_column: str
  label_column: str
  fieldnames: Optional[Sequence[str]] = None
  delimiter: str = ","
  quotechar: str = '"'


class Dataset(classification_dataset.ClassificationDataset):
  """Dataset library for text classifier."""

  def __init__(
      self,
      dataset: tf.data.Dataset,
      label_names: List[str],
      tfrecord_cache_files: Optional[cache_files_lib.TFRecordCacheFiles] = None,
      size: Optional[int] = None,
  ):
    super().__init__(dataset, label_names, size)
    if not tfrecord_cache_files:
      tfrecord_cache_files = cache_files_lib.TFRecordCacheFiles(
          cache_prefix_filename="tfrecord", num_shards=1
      )
    self.tfrecord_cache_files = tfrecord_cache_files

  @classmethod
  def from_csv(
      cls,
      filename: str,
      csv_params: CSVParameters,
      shuffle: bool = True,
      cache_dir: Optional[str] = None,
      num_shards: int = 1,
  ) -> "Dataset":
    """Loads text with labels from a CSV file.

    Args:
      filename: Name of the CSV file.
      csv_params: Parameters used for reading the CSV file.
      shuffle: If True, randomly shuffle the data.
      cache_dir: Optional parameter to specify where to store the preprocessed
        dataset. Only used for BERT models.
      num_shards: Optional parameter for num shards of the preprocessed dataset.
        Note that using more than 1 shard will reorder the dataset. Only used
        for BERT models.

    Returns:
      Dataset containing (text, label) pairs and other related info.
    """
    if cache_dir is None:
      cache_dir = tempfile.mkdtemp()
    # calculate hash for cache based off of files
    hasher = hashlib.md5()
    hasher.update(os.path.basename(filename).encode("utf-8"))
    with tf.io.gfile.GFile(filename, "r") as f:
      reader = csv.DictReader(
          f,
          fieldnames=csv_params.fieldnames,
          delimiter=csv_params.delimiter,
          quotechar=csv_params.quotechar)

    lines = list(reader)
    for line in lines:
      hasher.update(str(line).encode("utf-8"))

    if shuffle:
      random.shuffle(lines)

    label_names = sorted(set([line[csv_params.label_column] for line in lines]))
    index_by_label = {label: index for index, label in enumerate(label_names)}

    texts = [line[csv_params.text_column] for line in lines]
    text_ds = tf.data.Dataset.from_tensor_slices(tf.cast(texts, tf.string))
    label_indices = [
        index_by_label[line[csv_params.label_column]] for line in lines
    ]
    label_index_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(label_indices, tf.int64)
    )
    text_label_ds = tf.data.Dataset.zip((text_ds, label_index_ds))

    hasher.update(str(num_shards).encode("utf-8"))
    cache_prefix_filename = hasher.hexdigest()
    tfrecord_cache_files = cache_files_lib.TFRecordCacheFiles(
        cache_prefix_filename, cache_dir, num_shards
    )
    return Dataset(
        dataset=text_label_ds,
        label_names=label_names,
        tfrecord_cache_files=tfrecord_cache_files,
        size=len(texts),
    )
