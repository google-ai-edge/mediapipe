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
"""MediaPipe Text tasks shared library."""

import ctypes
import os

# resources dependency

_BASE_LIB_PATH = "mediapipe/tasks/c/"
_shared_lib = None


def load_shared_library():
  """Loads the shared library for text tasks."""
  global _shared_lib
  if _shared_lib is not None:
    return _shared_lib

  if os.name == "posix":  # Linux or macOS
    lib_path = _BASE_LIB_PATH + "libmediapipe.so"
  else:  # Windows
    lib_path = _BASE_LIB_PATH + "libmediapipe.dll"
  lib = ctypes.CDLL(resources.GetResourceFilename(lib_path))

  _shared_lib = lib
  return _shared_lib
