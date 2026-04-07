# Copyright 2025 The MediaPipe Authors.
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

"""C types for BaseOptions."""

import ctypes


class BaseOptionsC(ctypes.Structure):
  """C types for BaseOptions.

  Attributes:
    model_asset_buffer: `bytes`, the model asset buffer.
    model_asset_buffer_count: `int`, the size of the model asset buffer.
    model_asset_path: `bytes`, the path to the model asset.
    delegate: `int`, the delegate to use.
    host_environment: `int`, the environment in which the task is running.
    host_system: `int`, the system on which the task is running.
    host_version: `bytes`, the Python version as a UTF-8 string.
    ca_bundle_path: `bytes`, the path to the CA bundle file as a UTF-8 string.
  """
  _fields_ = [
      ("model_asset_buffer", ctypes.c_char_p),
      ("model_asset_buffer_count", ctypes.c_uint),
      ("model_asset_path", ctypes.c_char_p),
      ("delegate", ctypes.c_int),
      ("host_environment", ctypes.c_int),
      ("host_system", ctypes.c_int),
      ("host_version", ctypes.c_char_p),
      ("ca_bundle_path", ctypes.c_char_p),
  ]
