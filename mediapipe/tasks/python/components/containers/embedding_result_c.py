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
"""Embeddings C API types."""

import ctypes


class EmbeddingC(ctypes.Structure):
  """C struct for an embedding."""

  _fields_ = [
      ('float_embedding', ctypes.POINTER(ctypes.c_float)),
      ('quantized_embedding', ctypes.POINTER(ctypes.c_uint8)),
      ('values_count', ctypes.c_uint32),
      ('head_index', ctypes.c_int),
      ('head_name', ctypes.c_char_p),
  ]


class EmbeddingResultC(ctypes.Structure):
  """C struct for embedding result."""

  _fields_ = [
      ('embeddings', ctypes.POINTER(EmbeddingC)),
      ('embeddings_count', ctypes.c_uint32),
      ('has_timestamp_ms', ctypes.c_bool),
      ('timestamp_ms', ctypes.c_int64),
  ]
