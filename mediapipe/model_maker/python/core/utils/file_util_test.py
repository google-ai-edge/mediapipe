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
import os
import tempfile
from unittest import mock as unittest_mock

from absl.testing import absltest
import requests

from mediapipe.model_maker.python.core.utils import file_util


class FileUtilTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    mock_gettempdir = unittest_mock.patch.object(
        tempfile,
        'gettempdir',
        return_value=self.create_tempdir(),
        autospec=True,
    )
    self.mock_gettempdir = mock_gettempdir.start()
    self.addCleanup(mock_gettempdir.stop)

  def test_get_path(self):
    path = 'gesture_recognizer/hand_landmark_full.tflite'
    url = 'https://storage.googleapis.com/mediapipe-assets/hand_landmark_full.tflite'
    downloaded_files = file_util.DownloadedFiles(path, url, is_folder=False)
    model_path = downloaded_files.get_path()
    self.assertTrue(os.path.exists(model_path))
    self.assertGreater(os.path.getsize(model_path), 0)

  def test_get_path_folder(self):
    folder_contents = [
        'keras_metadata.pb',
        'saved_model.pb',
        'assets/vocab.txt',
        'variables/variables.data-00000-of-00001',
        'variables/variables.index',
    ]
    path = 'text_classifier/mobilebert_tiny'
    url = (
        'https://storage.googleapis.com/mediapipe-assets/mobilebert_tiny.tar.gz'
    )
    downloaded_files = file_util.DownloadedFiles(path, url, is_folder=True)
    model_path = downloaded_files.get_path()
    self.assertTrue(os.path.exists(model_path))
    for file_name in folder_contents:
      file_path = os.path.join(model_path, file_name)
      self.assertTrue(os.path.exists(file_path))
      self.assertGreater(os.path.getsize(file_path), 0)

  @unittest_mock.patch.object(requests, 'get', wraps=requests.get)
  def test_get_path_multiple_calls(self, mock_get):
    path = 'gesture_recognizer/hand_landmark_full.tflite'
    url = 'https://storage.googleapis.com/mediapipe-assets/hand_landmark_full.tflite'
    downloaded_files = file_util.DownloadedFiles(path, url, is_folder=False)
    model_path = downloaded_files.get_path()
    self.assertTrue(os.path.exists(model_path))
    self.assertGreater(os.path.getsize(model_path), 0)
    model_path_2 = downloaded_files.get_path()
    self.assertEqual(model_path, model_path_2)
    self.assertEqual(mock_get.call_count, 1)


if __name__ == '__main__':
  absltest.main()
