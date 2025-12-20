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
import platform
from typing import Any, List, Optional, Sequence

from importlib import resources
from mediapipe.tasks.python.core import mediapipe_c_utils
from mediapipe.tasks.python.core import serial_dispatcher

_BASE_LIB_PATH = 'mediapipe/tasks/c/'
_shared_lib = None
_CFunction = mediapipe_c_utils.CFunction


def load_raw_library(signatures: Sequence[_CFunction] = ()) -> ctypes.CDLL:
  """Loads the raw ctypes.CDLL shared library and registers signatures.

  This function loads the raw ctypes.CDLL shared library if it hasn't been
  loaded yet, otherwise it re-uses the existing instance. It attaches the
  provided signatures and makes them available across all callsites.

  Args:
    signatures: The ctypes function signatures to register in the library.

  Returns:
    The ctypes shared library.
  """
  global _shared_lib
  if _shared_lib is None:
    if os.name == 'posix':
      if platform.system() == 'Darwin':  # macOS
        lib_filename = 'libmediapipe.dylib'
      else:  # Linux
        lib_filename = 'libmediapipe.so'
    else:  # Windows
      lib_filename = 'libmediapipe.dll'
    lib_path_context = resources.files('mediapipe.tasks.c')
    absolute_lib_path = str(lib_path_context / lib_filename)
    _shared_lib = ctypes.CDLL(absolute_lib_path)

  for signature in signatures:
    c_func = getattr(_shared_lib, signature.func_name)
    c_func.argtypes = signature.argtypes
    c_func.restype = signature.restype

  # Register "MpErrorFree()"
  _shared_lib.MpErrorFree.argtypes = [ctypes.c_void_p]
  _shared_lib.MpErrorFree.restype = None

  return _shared_lib


def load_shared_library(
    signatures: Sequence[_CFunction] = (),
) -> serial_dispatcher.SerialDispatcher:
  """Loads the shared library in a SerialDispatcher and registers signatures.

  This function creates a thread-safe wrapper for the shared library that
  dispatches all calls through a single dedicated thread. Every call to this
  function uses a different dispatch library with its own thread.

  Args:
    signatures: The ctypes function signatures to register in the library.

  Returns:
    A thread-safe wrapper for the ctypes shared library.
  """
  raw_lib = load_raw_library()
  return serial_dispatcher.SerialDispatcher(raw_lib, signatures)


def convert_strings_to_ctypes_array(
    str_list: Optional[List[str]],
) -> Optional[Any]:
  """Converts a list of Python strings to a ctypes array of c_char_p.

  Args:
      str_list: A list of strings, or None.

  Returns:
      A ctypes array of c_char_p, or None if the input is None or empty.
  """
  if not str_list:  # Handles None and empty list
    return None
  num_elements = len(str_list)
  c_array = (ctypes.c_char_p * num_elements)()
  encoded_strings = [s.encode('utf-8') for s in str_list]
  c_array[:] = encoded_strings

  return c_array
