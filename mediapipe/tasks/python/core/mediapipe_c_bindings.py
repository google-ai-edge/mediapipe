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

# resources dependency

_BASE_LIB_PATH = 'mediapipe/tasks/c/'
_shared_lib = None


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


def handle_status(status: int):
  """Checks the MpStatus and raises an error if not MP_OK."""
  if status == MpStatus.MP_OK:
    return
  elif status == MpStatus.MP_CANCELLED:
    raise RuntimeError('Cancelled')
  elif status == MpStatus.MP_UNKNOWN:
    raise RuntimeError('Unknown error')
  elif status == MpStatus.MP_INVALID_ARGUMENT:
    raise ValueError('Invalid argument')
  elif status == MpStatus.MP_DEADLINE_EXCEEDED:
    raise RuntimeError('Deadline exceeded')
  elif status == MpStatus.MP_NOT_FOUND:
    raise RuntimeError('Not found')
  elif status == MpStatus.MP_ALREADY_EXISTS:
    raise RuntimeError('Already exists')
  elif status == MpStatus.MP_PERMISSION_DENIED:
    raise RuntimeError('Permission denied')
  elif status == MpStatus.MP_RESOURCE_EXHAUSTED:
    raise RuntimeError('Resource exhausted')
  elif status == MpStatus.MP_FAILED_PRECONDITION:
    raise RuntimeError('Failed precondition')
  elif status == MpStatus.MP_ABORTED:
    raise RuntimeError('Aborted')
  elif status == MpStatus.MP_OUT_OF_RANGE:
    raise ValueError('Out of range')
  elif status == MpStatus.MP_UNIMPLEMENTED:
    raise RuntimeError('Unimplemented')
  elif status == MpStatus.MP_INTERNAL:
    raise RuntimeError('Internal error')
  elif status == MpStatus.MP_UNAVAILABLE:
    raise RuntimeError('Unavailable')
  elif status == MpStatus.MP_DATA_LOSS:
    raise RuntimeError('Data loss')
  elif status == MpStatus.MP_UNAUTHENTICATED:
    raise RuntimeError('Unauthenticated')
  else:
    raise RuntimeError(f'Unexpected status: {status}')


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


def load_shared_library():
  """Loads the shared library for text tasks."""
  global _shared_lib
  if _shared_lib is not None:
    return _shared_lib

  if os.name == 'posix':  # Linux or macOS
    lib_path = _BASE_LIB_PATH + 'libmediapipe.so'
  else:  # Windows
    lib_path = _BASE_LIB_PATH + 'libmediapipe.dll'
  lib = ctypes.CDLL(resources.GetResourceFilename(lib_path))

  _shared_lib = lib
  return _shared_lib
