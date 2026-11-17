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

"""C types for MpBaseOptions."""

import ctypes


class MpBaseOptionsC(ctypes.Structure):
  """C types for MpBaseOptions.

  Attributes:
    model_asset_buffer: `bytes`, the model asset buffer.
    model_asset_buffer_count: `int`, the size of the model asset buffer.
    model_asset_path: `bytes`, the path to the model asset.
    file_descriptor: `int`, the file descriptor of the model asset.
    delegate: `int`, the delegate to use.
    host_environment: `int`, the environment in which the task is running.
    host_system: `int`, the system on which the task is running.
    host_version: `bytes`, the Python version as a UTF-8 string.
    ca_bundle_path: `bytes`, the path to the CA bundle file as a UTF-8 string.
    app_id: `bytes`, optional application id as a UTF-8 string.
    app_version: `bytes`, optional application version as a UTF-8 string.
    npu_dispatch_library_dir: `bytes`, optional directory containing the LiteRT
      NPU dispatch libraries as a UTF-8 string.
    cpu_num_threads: `int`, number of CPU threads to use.
    cpu_use_xnnpack: `bool`, whether to use XNNPACK on CPU.
    gpu_cached_kernel_path: `bytes`, optional GPU kernel cache path.
    gpu_serialized_model_dir: `bytes`, optional GPU serialized model directory.
    gpu_model_token: `bytes`, optional GPU model cache token.
    gpu_backend: `int`, the GPU backend to use.
  """

  _fields_ = [
      ("model_asset_buffer", ctypes.c_char_p),
      ("model_asset_buffer_count", ctypes.c_uint),
      ("model_asset_path", ctypes.c_char_p),
      ("file_descriptor", ctypes.c_int),
      ("delegate", ctypes.c_int),
      ("host_environment", ctypes.c_int),
      ("host_system", ctypes.c_int),
      ("host_version", ctypes.c_char_p),
      ("ca_bundle_path", ctypes.c_char_p),
      ("app_id", ctypes.c_char_p),
      ("app_version", ctypes.c_char_p),
      ("npu_dispatch_library_dir", ctypes.c_char_p),
      ("cpu_num_threads", ctypes.c_int),
      ("cpu_use_xnnpack", ctypes.c_bool),
      ("gpu_cached_kernel_path", ctypes.c_char_p),
      ("gpu_serialized_model_dir", ctypes.c_char_p),
      ("gpu_model_token", ctypes.c_char_p),
      ("gpu_backend", ctypes.c_int),
  ]
