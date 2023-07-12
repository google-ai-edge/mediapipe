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

import hashlib
import json
import os
import shutil
from unittest import mock as unittest_mock

import tensorflow as tf

from mediapipe.model_maker.python.vision.core import test_utils
from mediapipe.model_maker.python.vision.object_detector import dataset_util
from mediapipe.tasks.python.test import test_utils as tasks_test_utils


class DatasetUtilTest(tf.test.TestCase):

  def _assert_cache_files_equal(self, cf1, cf2):
    self.assertEqual(cf1.cache_prefix, cf2.cache_prefix)
    self.assertEqual(cf1.num_shards, cf2.num_shards)

  def _assert_cache_files_not_equal(self, cf1, cf2):
    self.assertNotEqual(cf1.cache_prefix, cf2.cache_prefix)

  def _get_cache_files_and_assert_neq_fn(self, cache_files_fn):
    def get_cache_files_and_assert_neq(cf, data_dir, cache_dir):
      new_cf = cache_files_fn(data_dir, cache_dir)
      self._assert_cache_files_not_equal(cf, new_cf)
      return new_cf

    return get_cache_files_and_assert_neq

  @unittest_mock.patch.object(hashlib, 'md5', autospec=True)
  def test_get_cache_files_coco(self, mock_md5):
    mock_md5.return_value.hexdigest.return_value = 'train'
    cache_files = dataset_util.get_cache_files_coco(
        tasks_test_utils.get_test_data_path('coco_data'), cache_dir='/tmp/'
    )
    self.assertEqual(cache_files.cache_prefix, '/tmp/train')
    self.assertLen(cache_files.tfrecord_files, 1)
    self.assertEqual(
        cache_files.tfrecord_files[0], '/tmp/train-00000-of-00001.tfrecord'
    )
    self.assertEqual(cache_files.metadata_file, '/tmp/train_metadata.yaml')

  def test_matching_get_cache_files_coco(self):
    cache_dir = self.create_tempdir()
    coco_folder = tasks_test_utils.get_test_data_path('coco_data')
    coco_folder_tmp = os.path.join(self.create_tempdir(), 'coco_data')
    shutil.copytree(coco_folder, coco_folder_tmp)
    cache_files1 = dataset_util.get_cache_files_coco(coco_folder, cache_dir)
    cache_files2 = dataset_util.get_cache_files_coco(coco_folder, cache_dir)
    self._assert_cache_files_equal(cache_files1, cache_files2)
    cache_files3 = dataset_util.get_cache_files_coco(coco_folder_tmp, cache_dir)
    self._assert_cache_files_equal(cache_files1, cache_files3)

  def test_not_matching_get_cache_files_coco(self):
    cache_dir = self.create_tempdir()
    temp_dir = self.create_tempdir()
    coco_folder = os.path.join(temp_dir, 'coco_data')
    shutil.copytree(
        tasks_test_utils.get_test_data_path('coco_data'), coco_folder
    )
    prev_cache_file = dataset_util.get_cache_files_coco(coco_folder, cache_dir)
    os.chmod(coco_folder, 0o700)
    os.chmod(os.path.join(coco_folder, 'images'), 0o700)
    os.chmod(os.path.join(coco_folder, 'labels.json'), 0o700)
    get_cache_files_and_assert_neq = self._get_cache_files_and_assert_neq_fn(
        dataset_util.get_cache_files_coco
    )
    # Test adding image
    test_utils.write_filled_jpeg_file(
        os.path.join(coco_folder, 'images', 'test.jpg'), [0, 0, 0], 50
    )
    prev_cache_file = get_cache_files_and_assert_neq(
        prev_cache_file, coco_folder, cache_dir
    )
    # Test modifying labels.json
    with open(os.path.join(coco_folder, 'labels.json'), 'w') as f:
      json.dump({'images': [{'id': 1, 'file_name': '000000000078.jpg'}]}, f)
    prev_cache_file = get_cache_files_and_assert_neq(
        prev_cache_file, coco_folder, cache_dir
    )

    # Test rename folder
    new_coco_folder = os.path.join(temp_dir, 'coco_data_renamed')
    shutil.move(coco_folder, new_coco_folder)
    coco_folder = new_coco_folder
    prev_cache_file = get_cache_files_and_assert_neq(
        prev_cache_file, new_coco_folder, cache_dir
    )

  @unittest_mock.patch.object(hashlib, 'md5', autospec=True)
  def test_get_cache_files_pascal_voc(self, mock_md5):
    mock_md5.return_value.hexdigest.return_value = 'train'
    cache_files = dataset_util.get_cache_files_pascal_voc(
        tasks_test_utils.get_test_data_path('pascal_voc_data'),
        cache_dir='/tmp/',
    )
    self.assertEqual(cache_files.cache_prefix, '/tmp/train')
    self.assertLen(cache_files.tfrecord_files, 1)
    self.assertEqual(
        cache_files.tfrecord_files[0], '/tmp/train-00000-of-00001.tfrecord'
    )
    self.assertEqual(cache_files.metadata_file, '/tmp/train_metadata.yaml')

  def test_matching_get_cache_files_pascal_voc(self):
    cache_dir = self.create_tempdir()
    pascal_folder = tasks_test_utils.get_test_data_path('pascal_voc_data')
    pascal_folder_temp = os.path.join(self.create_tempdir(), 'pascal_voc_data')
    shutil.copytree(pascal_folder, pascal_folder_temp)
    cache_files1 = dataset_util.get_cache_files_pascal_voc(
        pascal_folder, cache_dir
    )
    cache_files2 = dataset_util.get_cache_files_pascal_voc(
        pascal_folder, cache_dir
    )
    self._assert_cache_files_equal(cache_files1, cache_files2)
    cache_files3 = dataset_util.get_cache_files_pascal_voc(
        pascal_folder_temp, cache_dir
    )
    self._assert_cache_files_equal(cache_files1, cache_files3)

  def test_not_matching_get_cache_files_pascal_voc(self):
    cache_dir = self.create_tempdir()
    temp_dir = self.create_tempdir()
    pascal_folder = os.path.join(temp_dir, 'pascal_voc_data')
    shutil.copytree(
        tasks_test_utils.get_test_data_path('pascal_voc_data'), pascal_folder
    )
    prev_cache_files = dataset_util.get_cache_files_pascal_voc(
        pascal_folder, cache_dir
    )
    os.chmod(pascal_folder, 0o700)
    os.chmod(os.path.join(pascal_folder, 'images'), 0o700)
    os.chmod(os.path.join(pascal_folder, 'Annotations'), 0o700)
    get_cache_files_and_assert_neq = self._get_cache_files_and_assert_neq_fn(
        dataset_util.get_cache_files_pascal_voc
    )
    # Test adding xml file
    with open(os.path.join(pascal_folder, 'Annotations', 'test.xml'), 'w') as f:
      f.write('test')
    prev_cache_files = get_cache_files_and_assert_neq(
        prev_cache_files, pascal_folder, cache_dir
    )

    # Test rename folder
    new_pascal_folder = os.path.join(temp_dir, 'pascal_voc_data_renamed')
    shutil.move(pascal_folder, new_pascal_folder)
    pascal_folder = new_pascal_folder
    prev_cache_files = get_cache_files_and_assert_neq(
        prev_cache_files, new_pascal_folder, cache_dir
    )

  def test_is_cached(self):
    tempdir = self.create_tempdir()
    cache_files = dataset_util.get_cache_files_coco(
        tasks_test_utils.get_test_data_path('coco_data'), cache_dir=tempdir
    )
    self.assertFalse(cache_files.is_cached())
    with open(cache_files.tfrecord_files[0], 'w') as f:
      f.write('test')
    self.assertFalse(cache_files.is_cached())
    with open(cache_files.metadata_file, 'w') as f:
      f.write('test')
    self.assertTrue(cache_files.is_cached())

  def test_get_label_map_coco(self):
    coco_dir = tasks_test_utils.get_test_data_path('coco_data')
    label_map = dataset_util.get_label_map_coco(coco_dir)
    all_keys = sorted(label_map.keys())
    self.assertEqual(all_keys[0], 0)
    self.assertEqual(all_keys[-1], 11)
    self.assertLen(all_keys, 12)

  def test_get_label_map_pascal_voc(self):
    pascal_dir = tasks_test_utils.get_test_data_path('pascal_voc_data')
    label_map = dataset_util.get_label_map_pascal_voc(pascal_dir)
    all_keys = sorted(label_map.keys())
    self.assertEqual(label_map[0], 'background')
    self.assertEqual(all_keys[0], 0)
    self.assertEqual(all_keys[-1], 2)
    self.assertLen(all_keys, 3)

  def _validate_cache_files(self, cache_files, expected_size):
    # Checks the TFRecord file
    self.assertTrue(os.path.isfile(cache_files.tfrecord_files[0]))
    self.assertGreater(os.path.getsize(cache_files.tfrecord_files[0]), 0)

    # Checks the metadata file
    self.assertTrue(os.path.isfile(cache_files.metadata_file))
    self.assertGreater(os.path.getsize(cache_files.metadata_file), 0)
    metadata_dict = cache_files.load_metadata()
    self.assertEqual(metadata_dict['size'], expected_size)

  def test_coco_cache_files_writer(self):
    tempdir = self.create_tempdir()
    coco_dir = tasks_test_utils.get_test_data_path('coco_data')
    label_map = dataset_util.get_label_map_coco(coco_dir)
    cache_writer = dataset_util.COCOCacheFilesWriter(label_map)
    cache_files = dataset_util.get_cache_files_coco(coco_dir, cache_dir=tempdir)
    cache_writer.write_files(cache_files, coco_dir)
    self._validate_cache_files(cache_files, 3)

  def test_pascal_voc_cache_files_writer(self):
    tempdir = self.create_tempdir()
    pascal_dir = tasks_test_utils.get_test_data_path('pascal_voc_data')
    label_map = dataset_util.get_label_map_pascal_voc(pascal_dir)
    cache_writer = dataset_util.PascalVocCacheFilesWriter(label_map)
    cache_files = dataset_util.get_cache_files_pascal_voc(
        pascal_dir, cache_dir=tempdir
    )
    cache_writer.write_files(cache_files, pascal_dir)
    self._validate_cache_files(cache_files, 4)


if __name__ == '__main__':
  tf.test.main()
