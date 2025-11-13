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

"""A thread-safe dispatcher for a ctypes."""

import concurrent.futures
import ctypes
import functools
import threading
from typing import Sequence

from mediapipe.tasks.python.core import mediapipe_c_utils


class SerialDispatcher:
  """A wrapper class for a ctypes.CDLL object that serializes all calls.

  This ensures that functions from a non-thread-safe C library are called
  sequentially from a single dedicated thread, preventing race conditions
  and segmentation faults.
  """

  # Enable dynamic attributes as we register methods on this class via
  # reflection. This allows these methods to be used with type checking.
  _HAS_DYNAMIC_ATTRIBUTES = True

  def __init__(
      self, lib: ctypes.CDLL, signatures: Sequence[mediapipe_c_utils.CFunction]
  ):
    """Initializes the SerialDispatcher.

    Args:
      lib: The ctypes.CDLL object to wrap.
      signatures: the CFunction objects specifying the functions to wrap.
    """
    self._lib = lib
    self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    self._lock = threading.Lock()
    self._is_closed = False

    for signature in signatures:
      self._register_signature(signature)

  def _register_signature(self, signature: mediapipe_c_utils.CFunction):
    """Registers a wrapped C function as a method on the SerialDispatcher.

    This method attaches a wrapper method to this class that dispatches all
    calls to the underlying C function through this class' executor. The
    methods are directly attached to the SerialDispatcher object, allowing them
    to be used transparently.

    Args:
      signature: The CFunction object specifying the C function to wrap and
        register.
    """
    c_func = getattr(self._lib, signature.func_name)
    c_func.argtypes = signature.argtypes
    c_func.restype = signature.restype

    @functools.wraps(c_func)
    def dispatcher_wrapper(*args, **kwargs):
      if self._is_closed or mediapipe_c_utils.is_shutdown():
        # Ignore calls after during Python shutdown (e.g. calls to free C++
        # resources, which might fail if the ctypes object is no longer loaded)
        # TODO: b/456183832 - Return 0 once all APIs return MpStatus.
        return None
      future = self._executor.submit(c_func, *args, **kwargs)
      return future.result()

    setattr(self, signature.func_name, dispatcher_wrapper)

  def close(self):
    """Shuts down the dispatcher and waits for pending tasks to complete."""
    with self._lock:
      if self._is_closed:
        return
      self._is_closed = True
    self._executor.shutdown(wait=True)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    del exc_type, exc_val, exc_tb
    self.close()
