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
"""Common TFRecord cache files library."""

import dataclasses
import os
import tempfile
from typing import Any, Mapping, Sequence

import tensorflow as tf
import yaml


# Suffix of the meta data file name.
METADATA_FILE_SUFFIX = '_metadata.yaml'


@dataclasses.dataclass(frozen=True)
class TFRecordCacheFiles:
  """TFRecordCacheFiles dataclass to store and load cached TFRecord files.

  Attributes:
    cache_prefix_filename: The cache prefix filename. This is usually provided
      as a hash of the original data source to avoid different data sources
      resulting in the same cache file.
    cache_dir: The cache directory to save TFRecord and metadata file. When
      cache_dir is None, a temporary folder will be created and will not be
      removed automatically after training which makes it can be used later.
    num_shards: Number of shards for output tfrecord files.
  """

  cache_prefix_filename: str = 'cache_prefix'
  cache_dir: str = dataclasses.field(default_factory=tempfile.mkdtemp)
  num_shards: int = 1

  def __post_init__(self):
    if not tf.io.gfile.exists(self.cache_dir):
      tf.io.gfile.makedirs(self.cache_dir)
    if not self.cache_prefix_filename:
      raise ValueError('cache_prefix_filename cannot be empty.')
    if self.num_shards <= 0:
      raise ValueError(
          f'num_shards must be greater than 0, got {self.num_shards}'
      )

  @property
  def cache_prefix(self) -> str:
    """The cache prefix including the cache directory and the cache prefix filename."""
    return os.path.join(self.cache_dir, self.cache_prefix_filename)

  @property
  def tfrecord_files(self) -> Sequence[str]:
    """The TFRecord files."""
    tfrecord_files = [
        self.cache_prefix + '-%05d-of-%05d.tfrecord' % (i, self.num_shards)
        for i in range(self.num_shards)
    ]
    return tfrecord_files

  @property
  def metadata_file(self) -> str:
    """The metadata file."""
    return self.cache_prefix + METADATA_FILE_SUFFIX

  def get_writers(self) -> Sequence[tf.io.TFRecordWriter]:
    """Gets an array of TFRecordWriter objects.

    Note that these writers should each be closed using .close() when done.

    Returns:
      Array of TFRecordWriter objects
    """
    return [tf.io.TFRecordWriter(path) for path in self.tfrecord_files]

  def save_metadata(self, metadata):
    """Writes metadata to file.

    Args:
      metadata: A dictionary of metadata content to write. Exact format is
        dependent on the specific dataset, but typically includes a 'size' and
        'label_names' entry.
    """
    with tf.io.gfile.GFile(self.metadata_file, 'w') as f:
      yaml.dump(metadata, f)

  def load_metadata(self) -> Mapping[Any, Any]:
    """Reads metadata from file.

    Returns:
      Dictionary object containing metadata
    """
    if not tf.io.gfile.exists(self.metadata_file):
      return {}
    with tf.io.gfile.GFile(self.metadata_file, 'r') as f:
      metadata = yaml.load(f, Loader=yaml.FullLoader)
    return metadata

  def is_cached(self) -> bool:
    """Checks whether this CacheFiles is already cached."""
    all_cached_files = list(self.tfrecord_files) + [self.metadata_file]
    return all(tf.io.gfile.exists(f) for f in all_cached_files)
