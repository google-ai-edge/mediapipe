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
import threading
from typing import Any, Sequence

from mediapipe.tasks.python.core import mediapipe_c_utils


class SerialDispatcher:
  """A wrapper class for a ctypes.CDLL object that serializes all calls.

  This ensures that functions from a non-thread-safe C library are called
  sequentially from a single dedicated thread, preventing race conditions
  and segmentation faults.

  If a function is a CStatusFunction, the dispatcher will raise a Python
  exception if the returned MpStatus code is not kMpOk.
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
    # Guards the two flags below and makes every call atomic with respect to
    # them. Calls are already serialized by the single-threaded executor, so
    # holding this lock for the duration of a call costs no concurrency.
    self._lock = threading.Lock()
    # True once calls are no longer forwarded to the C library.
    self._is_closed = False
    # True once the executor has been shut down.
    self._is_shut_down = False

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
    handler = signature.create_python_wrapper(self._lib, self._executor)
    # The Mp<Task>Close() functions free the native task whose handle every
    # other call receives as its first argument, so no call may reach the C
    # library after one has run (use-after-free), and it must not run twice
    # (double free). Mp<Task>CloseResult() frees a result, not the task.
    closes_handle = signature.func_name.endswith('Close')

    def shutdown_aware_handler(*args, **kwargs) -> Any:
      # The closed check and the call itself must be atomic: if the lock were
      # released in between, a call that had passed the check could be queued
      # behind a concurrent close() and run on a freed handle.
      with self._lock:
        if self._is_closed:
          return
        if closes_handle:
          self._is_closed = True
        return handler(*args, **kwargs)

    setattr(self, signature.func_name, shutdown_aware_handler)

  def close(self):
    """Shuts down the dispatcher and waits for pending tasks to complete."""
    with self._lock:
      self._is_closed = True
      if self._is_shut_down:
        return
      self._is_shut_down = True
    self._executor.shutdown(wait=True)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    del exc_type, exc_val, exc_tb
    self.close()
