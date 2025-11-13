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
import enum
import os
from typing import Any, List, Optional, Sequence

# resources dependency
from mediapipe.tasks.python.core import mediapipe_c_utils
from mediapipe.tasks.python.core import serial_dispatcher

_BASE_LIB_PATH = 'mediapipe/tasks/c/'
_shared_lib = None
CFunction = mediapipe_c_utils.CFunction


class MpStatus(enum.IntEnum):
  """Status codes for MediaPipe C API functions."""

  MP_OK = 0
  MP_CANCELLED = 1
  MP_UNKNOWN = 2
  MP_INVALID_ARGUMENT = 3
  MP_DEADLINE_EXCEEDED = 4
  MP_NOT_FOUND = 5
  MP_ALREADY_EXISTS = 6
  MP_PERMISSION_DENIED = 7
  MP_RESOURCE_EXHAUSTED = 8
  MP_FAILED_PRECONDITION = 9
  MP_ABORTED = 10
  MP_OUT_OF_RANGE = 11
  MP_UNIMPLEMENTED = 12
  MP_INTERNAL = 13
  MP_UNAVAILABLE = 14
  MP_DATA_LOSS = 15
  MP_UNAUTHENTICATED = 16


def convert_to_exception(status: int) -> Exception | None:
  """Returns an exception based on the MpStatus code, or None if MP_OK."""
  match status:
    case MpStatus.MP_OK:
      return None
    case MpStatus.MP_CANCELLED:
      raise TimeoutError('Cancelled')
    case MpStatus.MP_UNKNOWN:
      return RuntimeError('Unknown error')
    case MpStatus.MP_INVALID_ARGUMENT:
      return ValueError('Invalid argument')
    case MpStatus.MP_DEADLINE_EXCEEDED:
      raise TimeoutError('Deadline exceeded')
    case MpStatus.MP_NOT_FOUND:
      raise FileNotFoundError('Not found')
    case MpStatus.MP_ALREADY_EXISTS:
      raise FileExistsError('Already exists')
    case MpStatus.MP_PERMISSION_DENIED:
      raise PermissionError('Permission denied')
    case MpStatus.MP_RESOURCE_EXHAUSTED:
      return RuntimeError('Resource exhausted')
    case MpStatus.MP_FAILED_PRECONDITION:
      return RuntimeError('Failed precondition')
    case MpStatus.MP_ABORTED:
      return RuntimeError('Aborted')
    case MpStatus.MP_OUT_OF_RANGE:
      raise IndexError('Out of range')
    case MpStatus.MP_UNIMPLEMENTED:
      raise NotImplementedError('Unimplemented')
    case MpStatus.MP_INTERNAL:
      return RuntimeError('Internal error')
    case MpStatus.MP_UNAVAILABLE:
      raise ConnectionError('Unavailable')
    case MpStatus.MP_DATA_LOSS:
      return RuntimeError('Data loss')
    case MpStatus.MP_UNAUTHENTICATED:
      raise PermissionError('Unauthenticated')
    case _:
      return RuntimeError(f'Unexpected status: {status}')


def handle_status(status: int):
  """Checks the MpStatus and raises an error if not MP_OK."""
  exception = convert_to_exception(status)
  if exception:
    raise exception


def handle_return_code(
    return_code: int, error_msg_prefix: str, error_msg: ctypes.c_char_p
):
  """Checks the return code and raises an error if not 0."""
  if return_code == 0:
    return
  elif error_msg.value is not None:
    error_message = error_msg.value.decode('utf-8')
    raise RuntimeError(f'{error_msg_prefix}: {error_message}')
  else:
    raise RuntimeError(
        f'{error_msg_prefix}: Unexpected return code {return_code}'
    )


def load_raw_library(signatures: Sequence[CFunction] = ()) -> ctypes.CDLL:
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
    if os.name == 'posix':  # Linux or macOS
      lib_path = _BASE_LIB_PATH + 'libmediapipe.so'
    else:  # Windows
      lib_path = _BASE_LIB_PATH + 'libmediapipe.dll'
    _shared_lib = ctypes.CDLL(resources.GetResourceFilename(lib_path))

  for signature in signatures:
    c_func = getattr(_shared_lib, signature.func_name)
    c_func.argtypes = signature.argtypes
    c_func.restype = signature.restype

  return _shared_lib


def load_shared_library(
    signatures: Sequence[CFunction] = (),
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
