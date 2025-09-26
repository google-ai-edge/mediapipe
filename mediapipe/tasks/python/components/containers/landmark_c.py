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

"""C types for Landmark."""

import ctypes


class LandmarkC(ctypes.Structure):
  """The ctypes struct for Landmark."""

  _fields_ = [
      ('x', ctypes.c_float),
      ('y', ctypes.c_float),
      ('z', ctypes.c_float),
      ('has_visibility', ctypes.c_bool),
      ('visibility', ctypes.c_float),
      ('has_presence', ctypes.c_bool),
      ('presence', ctypes.c_float),
      ('name', ctypes.c_char_p),
  ]


class LandmarksC(ctypes.Structure):
  """The ctypes struct for a list of Landmarks."""

  _fields_ = [
      ('landmarks', ctypes.POINTER(LandmarkC)),
      ('landmarks_count', ctypes.c_uint32),
  ]


class NormalizedLandmarkC(ctypes.Structure):
  """The ctypes struct for NormalizedLandmark."""

  _fields_ = [
      ('x', ctypes.c_float),
      ('y', ctypes.c_float),
      ('z', ctypes.c_float),
      ('has_visibility', ctypes.c_bool),
      ('visibility', ctypes.c_float),
      ('has_presence', ctypes.c_bool),
      ('presence', ctypes.c_float),
      ('name', ctypes.c_char_p),
  ]


class NormalizedLandmarksC(ctypes.Structure):
  """The ctypes struct for a list of NormalizedLandmarks."""

  _fields_ = [
      ('landmarks', ctypes.POINTER(NormalizedLandmarkC)),
      ('landmarks_count', ctypes.c_uint32),
  ]
