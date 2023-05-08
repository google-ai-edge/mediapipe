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
"""Object detector dataset library."""

from typing import Optional

import tensorflow as tf
import yaml

from mediapipe.model_maker.python.core.data import classification_dataset
from mediapipe.model_maker.python.vision.object_detector import dataset_util
from official.vision.dataloaders import tf_example_decoder


class Dataset(classification_dataset.ClassificationDataset):
  """Dataset library for object detector."""

  @classmethod
  def from_coco_folder(
      cls,
      data_dir: str,
      max_num_images: Optional[int] = None,
      cache_dir: Optional[str] = None,
  ) -> 'Dataset':
    """Loads images and labels from the given directory in COCO format.

    - https://cocodataset.org/#home

    Folder structure should be:
      <data_dir>/
        images/
          <file0>.jpg
          ...
        labels.json

    The `labels.json` annotations file should should have the following format:
    {
        "categories": [{"id": 0, "name": "background"}, ...],
        "images": [{"id": 0, "file_name": "<file0>.jpg"}, ...],
        "annotations": [{
           "id": 0,
           "image_id": 0,
           "category_id": 2,
           "bbox": [x-top left, y-top left, width, height],
           }, ...]
    }
    Note that category id 0 is reserved for the "background" class. It is
    optional to include, but if included it must be set to "background".


    Args:
      data_dir: Name of the directory containing the data files.
      max_num_images: Max number of images to process.
      cache_dir: The cache directory to save TFRecord and metadata files. The
        TFRecord files are a standardized format for training object detection
        while the metadata file is used to store information like dataset size
        and label mapping of id to label name. If the cache_dir is not set, a
        temporary folder will be created and will not be removed automatically
        after training which means it can be reused later.

    Returns:
      Dataset containing images and labels and other related info.
    Raises:
      ValueError: If the input data directory is empty.
      ValueError: If the label_name for id 0 is set to something other than
        the 'background' class.
    """
    cache_files = dataset_util.get_cache_files_coco(data_dir, cache_dir)
    if not dataset_util.is_cached(cache_files):
      label_map = dataset_util.get_label_map_coco(data_dir)
      cache_writer = dataset_util.COCOCacheFilesWriter(
          label_map=label_map, max_num_images=max_num_images
      )
      cache_writer.write_files(cache_files, data_dir)
    return cls.from_cache(cache_files.cache_prefix)

  @classmethod
  def from_pascal_voc_folder(
      cls,
      data_dir: str,
      max_num_images: Optional[int] = None,
      cache_dir: Optional[str] = None,
  ) -> 'Dataset':
    """Loads images and labels from the given directory in PASCAL VOC format.

    - http://host.robots.ox.ac.uk/pascal/VOC.

    Folder structure should be:
      <data_dir>/
        images/
          <file0>.jpg
          ...
        Annotations/
          <file0>.xml
          ...
    Each <file0>.xml annotation file should have the following format:
      <annotation>
        <filename>file0.jpg</filename>
        <object>
          <name>kangaroo</name>
          <bndbox>
            <xmin>233</xmin>
            <ymin>89</ymin>
            <xmax>386</xmax>
            <ymax>262</ymax>
          </bndbox>
        </object>
        <object>...</object>
      </annotation>

    Args:
      data_dir: Name of the directory containing the data files.
      max_num_images: Max number of images to process.
      cache_dir: The cache directory to save TFRecord and metadata files. The
        TFRecord files are a standardized format for training object detection
        while the metadata file is used to store information like dataset size
        and label mapping of id to label name. If the cache_dir is not set, a
        temporary folder will be created and will not be removed automatically
        after training which means it can be reused later.

    Returns:
      Dataset containing images and labels and other related info.
    Raises:
      ValueError: if the input data directory is empty.
    """
    cache_files = dataset_util.get_cache_files_pascal_voc(data_dir, cache_dir)
    if not dataset_util.is_cached(cache_files):
      label_map = dataset_util.get_label_map_pascal_voc(data_dir)
      cache_writer = dataset_util.PascalVocCacheFilesWriter(
          label_map=label_map, max_num_images=max_num_images
      )
      cache_writer.write_files(cache_files, data_dir)

    return cls.from_cache(cache_files.cache_prefix)

  @classmethod
  def from_cache(cls, cache_prefix: str) -> 'Dataset':
    """Loads the TFRecord data from cache.

    Args:
      cache_prefix: The cache prefix including the cache directory and the cache
        prefix filename, e.g: '/tmp/cache/train'.

    Returns:
      ObjectDetectorDataset object.
    """
    # Get TFRecord Files
    tfrecord_file_pattern = cache_prefix + '*.tfrecord'
    matched_files = tf.io.gfile.glob(tfrecord_file_pattern)
    if not matched_files:
      raise ValueError('TFRecord files are empty.')

    # Load meta_data.
    meta_data_file = cache_prefix + dataset_util.META_DATA_FILE_SUFFIX
    if not tf.io.gfile.exists(meta_data_file):
      raise ValueError("Metadata file %s doesn't exist." % meta_data_file)
    with tf.io.gfile.GFile(meta_data_file, 'r') as f:
      meta_data = yaml.load(f, Loader=yaml.FullLoader)

    dataset = tf.data.TFRecordDataset(matched_files)
    decoder = tf_example_decoder.TfExampleDecoder(regenerate_source_id=False)
    dataset = dataset.map(decoder.decode, num_parallel_calls=tf.data.AUTOTUNE)

    label_map = meta_data['label_map']
    label_names = [label_map[k] for k in sorted(label_map.keys())]

    return Dataset(
        dataset=dataset, size=meta_data['size'], label_names=label_names
    )
