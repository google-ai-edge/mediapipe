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

import mediapipe.tasks.python.core.base_options as base_options_module


class BaseOptionsC(ctypes.Structure):
  _fields_ = [
      ("model_asset_buffer", ctypes.c_char_p),
      ("model_asset_buffer_count", ctypes.c_uint),
      ("model_asset_path", ctypes.c_char_p),
  ]


def create_base_options_c(
    src: base_options_module.BaseOptions,
) -> BaseOptionsC:
  """Creates a BaseOptionsC struct."""
  options = BaseOptionsC()
  options.model_asset_buffer = src.model_asset_buffer
  options.model_asset_buffer_count = (
      len(src.model_asset_buffer) if src.model_asset_buffer else 0
  )
  options.model_asset_path = (
      src.model_asset_path.encode("utf-8") if src.model_asset_path else None
  )
  return options
