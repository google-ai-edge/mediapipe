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

import ctypes

from absl.testing import absltest

from mediapipe.tasks.python.core import base_options as base_options_lib
from mediapipe.tasks.python.core import base_options_c as base_options_c_lib


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
    self.assertEqual(options_c.file_descriptor, -1)

  def test_convert_to_ctypes_with_model_asset_buffer(self):
    options = base_options_lib.BaseOptions(model_asset_buffer=b'buffer')
    options_c = options.to_ctypes()
    self.assertIsNone(options_c.model_asset_path)
    self.assertEqual(options_c.model_asset_buffer, b'buffer')
    self.assertEqual(options_c.model_asset_buffer_count, 6)
    self.assertEqual(
        options_c.delegate, base_options_lib.BaseOptions.Delegate.CPU
    )
    self.assertEqual(options_c.file_descriptor, -1)

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
    self.assertEqual(options_c.file_descriptor, -1)

  def test_convert_to_ctypes_with_npu_delegate(self):
    options = base_options_lib.BaseOptions(
        model_asset_path='/path/to/model',
        delegate=base_options_lib.BaseOptions.Delegate.NPU,
        npu_dispatch_library_dir='/data/app/lib/arm64',
    )
    options_c = options.to_ctypes()
    self.assertEqual(options_c.model_asset_path, b'/path/to/model')
    self.assertEqual(
        options_c.delegate, base_options_lib.BaseOptions.Delegate.NPU
    )
    self.assertEqual(options_c.npu_dispatch_library_dir, b'/data/app/lib/arm64')
    self.assertEqual(options_c.file_descriptor, -1)

  def test_convert_to_ctypes_without_delegate(self):
    options = base_options_lib.BaseOptions(
        model_asset_path='/path/to/model', delegate=None
    )
    options_c = options.to_ctypes()
    self.assertEqual(options_c.model_asset_path, b'/path/to/model')
    self.assertEqual(
        options_c.delegate, base_options_lib.BaseOptions.Delegate.CPU
    )
    self.assertEqual(options_c.file_descriptor, -1)

  def test_c_struct_layout_and_zero_initialization(self):
    self.assertLen(base_options_c_lib.MpBaseOptionsC._fields_, 21)
    if ctypes.sizeof(ctypes.c_void_p) == 8:
      self.assertEqual(ctypes.sizeof(base_options_c_lib.MpBaseOptionsC), 144)
    options_c = base_options_lib.BaseOptions(
        model_asset_path='/path/to/model'
    ).to_ctypes()
    trailing_size = 2 * ctypes.sizeof(ctypes.c_size_t) + ctypes.sizeof(
        ctypes.c_void_p
    )
    raw_bytes = ctypes.string_at(
        ctypes.addressof(options_c), ctypes.sizeof(options_c)
    )
    self.assertEqual(raw_bytes[-trailing_size:], b'\x00' * trailing_size)
    self.assertEqual(options_c.litert_runtime_handle, 0)
    self.assertEqual(options_c.litert_gpu_accelerator_handle, 0)
    self.assertIsNone(options_c.litert_runtime_context)

  def test_equality(self):
    options1 = base_options_lib.BaseOptions(
        model_asset_path='/path/to/model',
        delegate=base_options_lib.BaseOptions.Delegate.CPU,
    )
    options2 = base_options_lib.BaseOptions(
        model_asset_path='/path/to/model',
        delegate=base_options_lib.BaseOptions.Delegate.CPU,
    )
    options3 = base_options_lib.BaseOptions(
        model_asset_path='/path/to/model',
        delegate=base_options_lib.BaseOptions.Delegate.GPU,
    )
    self.assertEqual(options1, options2)
    self.assertNotEqual(options1, options3)


if __name__ == '__main__':
  absltest.main()
