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

"""A thread-safe dispatcher for ctypes shared libraries."""

import ctypes
import functools
import queue
import threading
from typing import Any, Callable, Optional, Sequence
import weakref

from mediapipe.tasks.python.core import mediapipe_c_utils


def _worker_loop(
    owner_ref: weakref.ReferenceType["_SerialWorker"],
    work_queue: queue.SimpleQueue[
        Optional[tuple[Callable[[], Any], threading.Event, list[Any]]]
    ],
) -> None:
  """Executes queued callables serially until a None sentinel is received."""
  while True:
    item = work_queue.get()
    if item is None:
      del owner_ref  # Kept alive on the worker stack until _on_gc or exit.
      return
    fn, done_event, box = item
    del item
    try:
      box[0] = fn()
    except BaseException as exc:  # pylint: disable=broad-exception-caught
      box[1] = exc
    finally:
      # Drop the references to the call before waking up the caller, so that
      # objects the caller releases afterwards are not finalized on this thread.
      del fn, box
      done_event.set()
      del done_event


class _SerialWorker:
  """Executes callables sequentially on a dedicated daemon worker thread.

  Unlike `concurrent.futures.ThreadPoolExecutor`, this worker does not acquire
  any process-global shutdown lock during submission and falls back to direct
  execution if cyclic garbage collection finalized the weakref to this worker
  before a task's `__del__` method called `close()`.
  """

  def __init__(self):
    self._work_queue: queue.SimpleQueue[
        Optional[tuple[Callable[[], Any], threading.Event, list[Any]]]
    ] = queue.SimpleQueue()
    self._lock = threading.Lock()
    self._is_closed = False
    self._abandoned = [False]

    abandoned = self._abandoned
    work_queue = self._work_queue

    def _on_gc(_):
      abandoned[0] = True
      work_queue.put(None)

    owner_ref = weakref.ref(self, _on_gc)
    self._thread = threading.Thread(
        target=_worker_loop,
        args=(owner_ref, work_queue),
        daemon=True,
    )
    self._thread.start()

  def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Executes `fn(*args, **kwargs)` on the worker thread."""
    if mediapipe_c_utils.is_shutdown():
      return None
    on_worker_thread = self._thread is threading.current_thread()
    done_event = threading.Event()
    box: list[Any] = [None, None]
    with self._lock:
      if self._is_closed:
        return None
      # A call made on the worker thread itself, for example by a callback or a
      # finalizer that runs during another call, cannot be queued: the worker
      # would wait for itself.
      run_directly = self._abandoned[0] or on_worker_thread
      if not run_directly:
        self._work_queue.put((lambda: fn(*args, **kwargs), done_event, box))

    if run_directly:
      if not on_worker_thread:
        self._thread.join()
      return fn(*args, **kwargs)

    done_event.wait()
    err = box[1]
    result = box[0]
    box.clear()
    if err is not None:
      try:
        raise err
      finally:
        del err
    return result

  def close(self) -> None:
    """Stops the worker thread and waits until pending calls completed."""
    with self._lock:
      if not self._is_closed:
        self._is_closed = True
        if not self._abandoned[0]:
          self._work_queue.put(None)
    if (
        not mediapipe_c_utils.is_shutdown()
        and self._thread is not threading.current_thread()
    ):
      self._thread.join()


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
    self._worker = _SerialWorker()

    for signature in signatures:
      self._register_signature(signature)

  def _register_signature(self, signature: mediapipe_c_utils.CFunction):
    """Registers a wrapped C function as a method on the SerialDispatcher.

    Args:
      signature: The CFunction object specifying the C function to wrap and
        register.
    """
    c_wrapper = signature.create_c_wrapper(self._lib)
    worker = self._worker

    @functools.wraps(c_wrapper)
    def shutdown_aware_handler(*args, **kwargs) -> Any:
      return worker.call(c_wrapper, *args, **kwargs)

    setattr(self, signature.func_name, shutdown_aware_handler)

  def close(self):
    """Shuts down the dispatcher and waits for pending tasks to complete."""
    self._worker.close()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    del exc_type, exc_val, exc_tb
    self.close()
