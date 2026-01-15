# Copyright 2026 The MediaPipe Authors.
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
"""Tests for BaseOptions conversion between Python and C."""

from absl.testing import absltest

from mediapipe.tasks.python.core import base_options as base_options_lib


class BaseOptionsTest(absltest.TestCase):

  def test_convert_to_ctypes_with_model_asset_path(self):
    options = base_options_lib.BaseOptions(model_asset_path='/path/to/model')
    options_c = options.to_ctypes()
    self.assertEqual(options_c.model_asset_path, b'/path/to/model')
    self.assertIsNone(options_c.model_asset_buffer)
    self.assertEqual(options_c.model_asset_buffer_count, 0)
    self.assertEqual(
        options_c.delegate, base_options_lib.BaseOptions.Delegate.CPU
    )

  def test_convert_to_ctypes_with_model_asset_buffer(self):
    options = base_options_lib.BaseOptions(model_asset_buffer=b'buffer')
    options_c = options.to_ctypes()
    self.assertIsNone(options_c.model_asset_path)
    self.assertEqual(options_c.model_asset_buffer, b'buffer')
    self.assertEqual(options_c.model_asset_buffer_count, 6)
    self.assertEqual(
        options_c.delegate, base_options_lib.BaseOptions.Delegate.CPU
    )

  def test_convert_to_ctypes_with_gpu_delegate(self):
    options = base_options_lib.BaseOptions(
        model_asset_path='/path/to/model',
        delegate=base_options_lib.BaseOptions.Delegate.GPU,
    )
    options_c = options.to_ctypes()
    self.assertEqual(options_c.model_asset_path, b'/path/to/model')
    self.assertEqual(
        options_c.delegate, base_options_lib.BaseOptions.Delegate.GPU
    )

  def test_convert_to_ctypes_without_delegate(self):
    options = base_options_lib.BaseOptions(
        model_asset_path='/path/to/model', delegate=None
    )
    options_c = options.to_ctypes()
    self.assertEqual(options_c.model_asset_path, b'/path/to/model')
    self.assertEqual(
        options_c.delegate, base_options_lib.BaseOptions.Delegate.CPU
    )


if __name__ == '__main__':
  absltest.main()
