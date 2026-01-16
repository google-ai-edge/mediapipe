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
# ==============================================================================
"""Tests for model asset bundle utilities."""

import os
import tempfile
import zipfile

from absl.testing import absltest
from mediapipe.tasks.python.metadata.metadata_writers import model_asset_bundle_utils


class ModelAssetBundleUtilsTest(absltest.TestCase):

  def test_create_model_asset_bundle(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      bundle_file = os.path.join(temp_dir, 'test.task')
      input_models = {'1.tflite': b'\x11\x22', '2.tflite': b'\x33'}
      model_asset_bundle_utils.create_model_asset_bundle(
          input_models, bundle_file
      )
      with zipfile.ZipFile(bundle_file) as zf:
        for info in zf.infolist():
          # Each file should be aligned.
          header_length = len(info.FileHeader())
          offset = info.header_offset + header_length
          self.assertEqual(offset % model_asset_bundle_utils._ALIGNMENT, 0)


if __name__ == '__main__':
  absltest.main()
