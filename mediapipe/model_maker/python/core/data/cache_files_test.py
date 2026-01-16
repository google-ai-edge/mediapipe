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

import tensorflow as tf

from mediapipe.model_maker.python.core.data import cache_files


class CacheFilesTest(tf.test.TestCase):

  def test_tfrecord_cache_files(self):
    cf = cache_files.TFRecordCacheFiles(
        cache_prefix_filename='tfrecord',
        cache_dir='/tmp/cache_dir',
        num_shards=2,
    )
    self.assertEqual(cf.cache_prefix, '/tmp/cache_dir/tfrecord')
    self.assertEqual(
        cf.metadata_file,
        '/tmp/cache_dir/tfrecord' + cache_files.METADATA_FILE_SUFFIX,
    )
    expected_tfrecord_files = [
        '/tmp/cache_dir/tfrecord-%05d-of-%05d.tfrecord' % (i, 2)
        for i in range(2)
    ]
    self.assertEqual(cf.tfrecord_files, expected_tfrecord_files)

    # Writing TFRecord Files
    self.assertFalse(cf.is_cached())
    for tfrecord_file in cf.tfrecord_files:
      self.assertFalse(tf.io.gfile.exists(tfrecord_file))
    writers = cf.get_writers()
    for writer in writers:
      writer.close()
    for tfrecord_file in cf.tfrecord_files:
      self.assertTrue(tf.io.gfile.exists(tfrecord_file))
    self.assertFalse(cf.is_cached())

    # Writing Metadata Files
    original_metadata = {'size': 10, 'label_names': ['label1', 'label2']}
    cf.save_metadata(original_metadata)
    self.assertTrue(cf.is_cached())
    metadata = cf.load_metadata()
    self.assertEqual(metadata, original_metadata)

  def test_recordio_cache_files_error(self):
    with self.assertRaisesRegex(
        ValueError, 'cache_prefix_filename cannot be empty'
    ):
      cache_files.TFRecordCacheFiles(
          cache_prefix_filename='',
          cache_dir='/tmp/cache_dir',
          num_shards=2,
      )
    with self.assertRaisesRegex(
        ValueError, 'num_shards must be greater than 0, got 0'
    ):
      cache_files.TFRecordCacheFiles(
          cache_prefix_filename='tfrecord',
          cache_dir='/tmp/cache_dir',
          num_shards=0,
      )


if __name__ == '__main__':
  tf.test.main()
