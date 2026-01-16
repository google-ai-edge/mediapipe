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
"""MediaPipe Rect C API types."""


import ctypes


class RectC(ctypes.Structure):
  """A rectangle, used as part of detection results."""

  _fields_ = [
      ('left', ctypes.c_int),
      ('top', ctypes.c_int),
      ('bottom', ctypes.c_int),
      ('right', ctypes.c_int),
  ]


class RectFC(ctypes.Structure):
  """A rectangle (operating on floats), used as part of processing options."""

  _fields_ = [
      ('left', ctypes.c_float),
      ('top', ctypes.c_float),
      ('bottom', ctypes.c_float),
      ('right', ctypes.c_float),
  ]


class NormalizedRectC(ctypes.Structure):
  """A rectangle with rotation in normalized coordinates."""

  _fields_ = [
      ('left', ctypes.c_float),
      ('top', ctypes.c_float),
      ('bottom', ctypes.c_float),
      ('right', ctypes.c_float),
  ]
