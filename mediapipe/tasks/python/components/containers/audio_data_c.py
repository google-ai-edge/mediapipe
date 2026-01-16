# Copyright 2025 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MediaPipe audio data ctypes."""

import ctypes


class AudioDataC(ctypes.Structure):
  """The audio data used in the C API."""

  _fields_ = [
      ('num_channels', ctypes.c_int),
      ('sample_rate', ctypes.c_double),
      ('audio_data', ctypes.POINTER(ctypes.c_float)),
      ('audio_data_size', ctypes.c_size_t),
  ]
