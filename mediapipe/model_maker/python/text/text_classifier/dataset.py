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
import random

from typing import Optional, Sequence
import tensorflow as tf

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

  @classmethod
  def from_csv(cls,
               filename: str,
               csv_params: CSVParameters,
               shuffle: bool = True) -> "Dataset":
    """Loads text with labels from a CSV file.

    Args:
      filename: Name of the CSV file.
      csv_params: Parameters used for reading the CSV file.
      shuffle: If True, randomly shuffle the data.

    Returns:
      Dataset containing (text, label) pairs and other related info.
    """
    with tf.io.gfile.GFile(filename, "r") as f:
      reader = csv.DictReader(
          f,
          fieldnames=csv_params.fieldnames,
          delimiter=csv_params.delimiter,
          quotechar=csv_params.quotechar)

    lines = list(reader)
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
        tf.cast(label_indices, tf.int64))
    text_label_ds = tf.data.Dataset.zip((text_ds, label_index_ds))

    return Dataset(
        dataset=text_label_ds, size=len(texts), label_names=label_names)
