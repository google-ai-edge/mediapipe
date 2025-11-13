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
"""Common MediaPipe CTypes definitions and utilities."""

import atexit
import dataclasses
import threading
from typing import Any


# Global shutdown state to ignore calls during Python shutdown. This is used to
# prevent forwarding calls to the underlying executor, which prints warning logs
# to the console after shutdown.
_shutdown = False
_global_shutdown_lock = threading.Lock()


def _python_exit():
  global _shutdown
  with _global_shutdown_lock:
    _shutdown = True


def is_shutdown():
  with _global_shutdown_lock:
    return _shutdown


atexit.register(_python_exit)


@dataclasses.dataclass(frozen=True)
class CFunction:
  """Stores the ctypes signature information for a C function.

  Attributes:
    func_name: The name of the C function.
    argtypes: The ctypes types for the function arguments.
    restype: The ctypes type for the function's return value.
  """

  func_name: str
  argtypes: list[Any]
  restype: Any
