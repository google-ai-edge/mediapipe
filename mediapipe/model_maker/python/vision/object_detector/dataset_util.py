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
"""Utilities for Object Detector Dataset Library."""

import abc
import collections
import hashlib
import json
import math
import os
import tempfile
from typing import Any, Dict, List, Mapping, Optional
import xml.etree.ElementTree as ET

import tensorflow as tf

from mediapipe.model_maker.python.core.data import cache_files
from official.vision.data import tfrecord_lib


def _xml_get(node: ET.Element, name: str) -> ET.Element:
  """Gets a named child from an XML Element node.

  This method is used to retrieve an XML element that is expected to exist as a
  subelement of the `node` passed into this argument. If the subelement is not
  found, then an error is thrown.

  Raises:
    ValueError: If the subelement is not found.

  Args:
    node: XML Element Tree node.
    name: Name of the child node to get

  Returns:
    A child node of the parameter node with the matching name.
  """
  result = node.find(name)
  if result is None:
    raise ValueError(f'Unexpected xml format: {name} not found in {node}')
  return result


def _get_cache_dir_or_create(cache_dir: Optional[str]) -> str:
  """Gets the cache directory or creates it if not exists."""
  if cache_dir is None:
    cache_dir = tempfile.mkdtemp()
  if not tf.io.gfile.exists(cache_dir):
    tf.io.gfile.makedirs(cache_dir)
  return cache_dir


def _get_dir_basename(data_dir: str) -> str:
  """Gets the base name of the directory."""
  return os.path.basename(os.path.abspath(data_dir))


def _get_cache_files(
    cache_dir: Optional[str], cache_prefix_filename: str, num_shards: int = 10
) -> cache_files.TFRecordCacheFiles:
  """Creates an object of CacheFiles class.

  Args:
    cache_dir: The cache directory to save TFRecord and metadata file. When
      cache_dir is None, a temporary folder will be created and will not be
      removed automatically after training which makes it can be used later.
     cache_prefix_filename: The cache prefix filename.
     num_shards: Number of shards for output file.

  Returns:
    An object of CacheFiles class.
  """
  cache_dir = _get_cache_dir_or_create(cache_dir)
  return cache_files.TFRecordCacheFiles(
      cache_prefix_filename=cache_prefix_filename,
      cache_dir=cache_dir,
      num_shards=num_shards,
  )


def get_cache_files_coco(
    data_dir: str, cache_dir: str
) -> cache_files.TFRecordCacheFiles:
  """Creates an object of CacheFiles class using a COCO formatted dataset.

  Args:
    data_dir: Folder path of the coco dataset
    cache_dir: Folder path of the cache location. When cache_dir is None, a
      temporary folder will be created and will not be removed automatically
      after training which makes it can be used later.

  Returns:
    An object of CacheFiles class.
  """
  hasher = hashlib.md5()
  # Update with dataset folder name
  hasher.update(_get_dir_basename(data_dir).encode('utf-8'))
  # Update with image filenames
  for image_file in sorted(os.listdir(os.path.join(data_dir, 'images'))):
    hasher.update(os.path.basename(image_file).encode('utf-8'))
  # Update with labels.json file content
  label_file = os.path.join(data_dir, 'labels.json')
  with open(label_file, 'r') as f:
    label_json = json.load(f)
    hasher.update(str(label_json).encode('utf-8'))
    num_examples = len(label_json['images'])
  # Num_shards automatically set to 100 images per shard, up to 10 shards total.
  # See https://www.tensorflow.org/tutorials/load_data/tfrecord for more info
  # on sharding.
  num_shards = min(math.ceil(num_examples / 100), 10)
  # Update with num shards
  hasher.update(str(num_shards).encode('utf-8'))
  cache_prefix_filename = hasher.hexdigest()

  return _get_cache_files(cache_dir, cache_prefix_filename, num_shards)


def get_cache_files_pascal_voc(
    data_dir: str, cache_dir: str
) -> cache_files.TFRecordCacheFiles:
  """Gets an object of CacheFiles using a PASCAL VOC formatted dataset.

  Args:
    data_dir: Folder path of the pascal voc dataset.
    cache_dir: Folder path of the cache location. When cache_dir is None, a
      temporary folder will be created and will not be removed automatically
      after training which makes it can be used later.

  Returns:
    An object of CacheFiles class.
  """
  hasher = hashlib.md5()
  hasher.update(_get_dir_basename(data_dir).encode('utf-8'))
  annotation_files = tf.io.gfile.glob(
      os.path.join(data_dir, 'Annotations') + r'/*.xml'
  )
  annotation_filenames = [
      os.path.basename(ann_file) for ann_file in annotation_files
  ]
  hasher.update(' '.join(annotation_filenames).encode('utf-8'))
  num_examples = len(annotation_filenames)
  num_shards = min(math.ceil(num_examples / 100), 10)
  hasher.update(str(num_shards).encode('utf-8'))
  cache_prefix_filename = hasher.hexdigest()

  return _get_cache_files(cache_dir, cache_prefix_filename, num_shards)


class CacheFilesWriter(abc.ABC):
  """CacheFilesWriter class to write the cached files."""

  def __init__(
      self, label_map: Dict[int, str], max_num_images: Optional[int] = None
  ) -> None:
    """Initializes CacheFilesWriter for object detector.

    Args:
      label_map: Dict, map label integer ids to string label names such as {1:
        'person', 2: 'notperson'}. 0 is the reserved key for `background` and
        doesn't need to be included in `label_map`. Label names can't be
        duplicated.
      max_num_images: Max number of images to process. If None, process all the
        images.
    """
    self.label_map = label_map
    self.max_num_images = max_num_images

  def write_files(
      self,
      tfrecord_cache_files: cache_files.TFRecordCacheFiles,
      *args,
      **kwargs,
  ) -> None:
    """Writes TFRecord and metadata files.

    Args:
      tfrecord_cache_files: TFRecordCacheFiles object including a list of
        TFRecord files and the meta data yaml file to save the metadata
        including data size and label_map.
      *args: Non-keyword of parameters used in the `_get_example` method.
      **kwargs: Keyword parameters used in the `_get_example` method.
    """
    writers = tfrecord_cache_files.get_writers()

    # Writes tf.Example into TFRecord files.
    size = 0
    for idx, tf_example in enumerate(self._get_example(*args, **kwargs)):
      if self.max_num_images and idx >= self.max_num_images:
        break
      if idx % 100 == 0:
        tf.compat.v1.logging.info('On image %d' % idx)
      writers[idx % len(writers)].write(tf_example.SerializeToString())
      size = idx + 1

    for writer in writers:
      writer.close()

    # Writes metadata into metadata_file.
    metadata = {'size': size, 'label_map': self.label_map}
    tfrecord_cache_files.save_metadata(metadata)

  @abc.abstractmethod
  def _get_example(self, *args, **kwargs):
    raise NotImplementedError


def get_label_map_coco(data_dir: str):
  """Gets the label map from a COCO formatted dataset directory.

  Note that id 0 is reserved for the background class. If id=0 is set, it needs
  to be set to "background". It is optional to include id=0 if it is unused, and
  it will be automatically added by this method.

  Args:
    data_dir: Path of the dataset directory

  Returns:
    label_map dictionary of the format {<id>:<label_name>}

  Raises:
    ValueError: If the label_name for id 0 is set to something other than
    the "background" class.
  """
  data_dir = os.path.abspath(data_dir)
  # Process labels.json file
  label_file = os.path.join(data_dir, 'labels.json')
  with open(label_file, 'r') as f:
    data = json.load(f)

  # Categories
  label_map = {}
  for category in data['categories']:
    label_map[int(category['id'])] = category['name']

  if 0 in label_map and label_map[0] != 'background':
    raise ValueError(
        (
            'Label index 0 is reserved for the background class, but '
            f'it was found to be {label_map[0]}'
        ),
    )
  if 0 not in label_map:
    label_map[0] = 'background'

  return label_map


def get_label_map_pascal_voc(data_dir: str):
  """Gets the label map from a PASCAL VOC formatted dataset directory.

  The id to label_name mapping is determined by sorting all label_names and
  numbering them starting from 1. Id=0 is set as the 'background' class.

  Args:
    data_dir: Path of the dataset directory

  Returns:
    label_map dictionary of the format {<id>:<label_name>}
  """
  data_dir = os.path.abspath(data_dir)
  all_label_names = set()
  annotations_dir = os.path.join(data_dir, 'Annotations')
  all_annotation_files = tf.io.gfile.glob(annotations_dir + r'/*.xml')
  for ann_file in all_annotation_files:
    tree = ET.parse(ann_file)
    root = tree.getroot()
    for child in root.iter('object'):
      label_name = _xml_get(child, 'name').text
      all_label_names.add(label_name)
  label_map = {0: 'background'}
  for ind, label_name in enumerate(sorted(all_label_names)):
    label_map[ind + 1] = label_name
  return label_map


def _bbox_data_to_feature_dict(data):
  """Converts a dictionary of bbox annotations to a feature dictionary.

  Args:
    data: Dict with keys 'xmin', 'xmax', 'ymin', 'ymax', 'category_id'

  Returns:
    Feature dictionary
  """
  bbox_feature_dict = {
      'image/object/bbox/xmin': tfrecord_lib.convert_to_feature(data['xmin']),
      'image/object/bbox/xmax': tfrecord_lib.convert_to_feature(data['xmax']),
      'image/object/bbox/ymin': tfrecord_lib.convert_to_feature(data['ymin']),
      'image/object/bbox/ymax': tfrecord_lib.convert_to_feature(data['ymax']),
      'image/object/class/label': tfrecord_lib.convert_to_feature(
          data['category_id']
      ),
  }
  return bbox_feature_dict


def _coco_annotations_to_lists(
    bbox_annotations: List[Mapping[str, Any]],
    image_height: int,
    image_width: int,
):
  """Converts COCO annotations to feature lists.

  Args:
    bbox_annotations: List of dicts with keys ['bbox', 'category_id']
    image_height: Height of image
    image_width: Width of image

  Returns:
    (data, num_annotations_skipped) tuple where data contains the keys:
    ['xmin', 'xmax', 'ymin', 'ymax', 'is_crowd', 'category_id', 'area'] and
    num_annotations_skipped is the number of skipped annotations because of the
    bbox having 0 area.
  """

  data = collections.defaultdict(list)

  num_annotations_skipped = 0

  for object_annotations in bbox_annotations:
    (x, y, width, height) = tuple(object_annotations['bbox'])

    if width <= 0 or height <= 0:
      num_annotations_skipped += 1
      continue
    if x + width > image_width or y + height > image_height:
      num_annotations_skipped += 1
      continue
    data['xmin'].append(float(x) / image_width)
    data['xmax'].append(float(x + width) / image_width)
    data['ymin'].append(float(y) / image_height)
    data['ymax'].append(float(y + height) / image_height)
    category_id = int(object_annotations['category_id'])
    data['category_id'].append(category_id)

  return data, num_annotations_skipped


class COCOCacheFilesWriter(CacheFilesWriter):
  """CacheFilesWriter class to write the cached files for COCO data."""

  def _get_example(self, data_dir: str) -> tf.train.Example:
    """Iterates over all examples in the COCO formatted dataset directory.

    Args:
      data_dir: Path of the dataset directory

    Yields:
      tf.train.Example
    """
    data_dir = os.path.abspath(data_dir)
    # Process labels.json file
    label_file = os.path.join(data_dir, 'labels.json')
    with open(label_file, 'r') as f:
      data = json.load(f)

    # Load all Annotations
    img_to_annotations = collections.defaultdict(list)
    for annotation in data['annotations']:
      image_id = annotation['image_id']
      img_to_annotations[image_id].append(annotation)

    # For each Image:
    for image in data['images']:
      img_id = image['id']
      file_name = image['file_name']
      full_path = os.path.join(data_dir, 'images', file_name)
      with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
      image = tf.io.decode_jpeg(encoded_jpg, channels=3)
      height, width, _ = image.shape
      feature_dict = tfrecord_lib.image_info_to_feature_dict(
          height, width, file_name, img_id, encoded_jpg, 'jpg'
      )
      data, _ = _coco_annotations_to_lists(
          img_to_annotations[img_id], height, width
      )
      if not data['xmin']:
        # Skip examples which have no annotations
        continue
      bbox_feature_dict = _bbox_data_to_feature_dict(data)
      feature_dict.update(bbox_feature_dict)
      example = tf.train.Example(
          features=tf.train.Features(feature=feature_dict)
      )
      yield example


class PascalVocCacheFilesWriter(CacheFilesWriter):
  """CacheFilesWriter class to write the cached files for PASCAL VOC data."""

  def _get_example(self, data_dir: str) -> tf.train.Example:
    """Iterates over all examples in the PASCAL VOC formatted dataset directory.

    Args:
      data_dir: Path of the dataset directory

    Yields:
      tf.train.Example
    """
    label_name_to_id = {name: i for (i, name) in self.label_map.items()}
    annotations_dir = os.path.join(data_dir, 'Annotations')
    images_dir = os.path.join(data_dir, 'images')
    all_annotation_paths = tf.io.gfile.glob(annotations_dir + r'/*.xml')

    for ind, ann_file in enumerate(all_annotation_paths):
      data = collections.defaultdict(list)
      tree = ET.parse(ann_file)
      root = tree.getroot()
      img_filename = _xml_get(root, 'filename').text
      img_file = os.path.join(images_dir, img_filename)
      with tf.io.gfile.GFile(img_file, 'rb') as fid:
        encoded_jpg = fid.read()
      image = tf.io.decode_jpeg(encoded_jpg, channels=3)
      height, width, _ = image.shape
      for child in root.iter('object'):
        category_name = _xml_get(child, 'name').text
        category_id = label_name_to_id[category_name]
        bndbox = _xml_get(child, 'bndbox')
        xmin = float(_xml_get(bndbox, 'xmin').text)
        xmax = float(_xml_get(bndbox, 'xmax').text)
        ymin = float(_xml_get(bndbox, 'ymin').text)
        ymax = float(_xml_get(bndbox, 'ymax').text)
        if xmax <= xmin or ymax <= ymin or xmax > width or ymax > height:
          # Skip annotations that have no area or are larger than the image
          continue
        data['xmin'].append(xmin / width)
        data['ymin'].append(ymin / height)
        data['xmax'].append(xmax / width)
        data['ymax'].append(ymax / height)
        data['category_id'].append(category_id)
      if not data['xmin']:
        # Skip examples which have no valid annotations
        continue
      feature_dict = tfrecord_lib.image_info_to_feature_dict(
          height, width, img_filename, ind, encoded_jpg, 'jpg'
      )
      bbox_feature_dict = _bbox_data_to_feature_dict(data)
      feature_dict.update(bbox_feature_dict)
      example = tf.train.Example(
          features=tf.train.Features(feature=feature_dict)
      )
      yield example
