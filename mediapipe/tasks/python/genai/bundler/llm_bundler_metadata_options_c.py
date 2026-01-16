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
"""Ctypes wrapper for LlmBundlerMetadataOptions."""

import ctypes


class LlmBundlerMetadataOptionsC(ctypes.Structure):
  """Wrapper for LlmBundlerMetadataOptions in C."""

  _fields_ = [
      ("start_token", ctypes.c_char_p),
      ("stop_tokens", ctypes.POINTER(ctypes.c_char_p)),
      ("num_stop_tokens", ctypes.c_int),
      ("enable_bytes_to_unicode_mapping", ctypes.c_bool),
      ("system_prompt", ctypes.c_char_p),
      ("prompt_prefix_user", ctypes.c_char_p),
      ("prompt_suffix_user", ctypes.c_char_p),
      ("prompt_prefix_model", ctypes.c_char_p),
      ("prompt_suffix_model", ctypes.c_char_p),
      ("prompt_prefix_system", ctypes.c_char_p),
      ("prompt_suffix_system", ctypes.c_char_p),
      ("user_role_token", ctypes.c_char_p),
      ("system_role_token", ctypes.c_char_p),
      ("model_role_token", ctypes.c_char_p),
      ("end_role_token", ctypes.c_char_p),
  ]
