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

"""A dispatcher to handle asynchronous results from a C library."""

import ctypes
import dataclasses
import enum
import logging
import os
import queue
import threading
from typing import Any, Callable, Optional, TypeVar, Union

from mediapipe.tasks.python.core import mediapipe_c_utils

CCallbackType = TypeVar("CCallbackType", bound=Callable[..., Any])


@dataclasses.dataclass(frozen=True)
class LiveStreamPacket:
  """A packet holding the contents of an async stream packet.

  Attributes:
    contents: The data that will be passed to the user-provided callback.
  """

  contents: tuple[object, ...]


@dataclasses.dataclass(frozen=True)
class _ExceptionPacket:
  """A packet holding an exception during an async operation.

  Attributes:
    exception: The exception that will be logged in the dispatcher thread.
  """

  exception: Exception


_AsyncResultPacket = Union[LiveStreamPacket, _ExceptionPacket]


class _DispatcherState(enum.Enum):
  """The state of the AsyncResultDispatcher."""

  NOT_STARTED = 0
  RUNNING = 1
  SHUTTING_DOWN = 2


_PIPE_NOT_INITIALIZED = -1


class AsyncResultDispatcher:
  """A dispatcher to safely handle asynchronous results from a C library.

  This class ensures that the user-provided Python callback is executed on a
  dedicated, GIL-holding Python thread, rather than directly on the C library's
  background thread. To use this class, you must supply a converter function
  that converts the C data types and then pass the Python callback you want to
  invoke to `wrap_callback()`, which returns a C callback that can be passed
  to the C library.

  The implementation uses OS level pipes to reduce the amount of Python code
  that is run without the GIL lock. The dispatcher trhead is only started when
  the C callback is first called.

  Here's an example of how to use `AsyncResultDispatcher`:

  ```
  def python_callback(result_data: MyResultType, timestamp: int):
      print(f"Received result at {timestamp}: {result_data}")

  def converter(c_result_ptr: ctypes.POINTER(CResult), c_timestamp: int):
      py_result = MyResultType.from_ctypes(c_result_ptr[0])
      py_timestamp = c_timestamp
      return py_result, py_timestamp

  dispatcher = AsyncResultDispatcher(converter=converter)
  c_callback_type = ctypes.CFUNCTYPE(
      None,  # Return type (void)
      ctypes.c_void_p,  # User data
      ctypes.c_int32,  # Status code
      ctypes.c_char_p,  # Error message
      ctypes.POINTER(CResult), # First argument
      ctypes.c_int64 # Second argument
  )
  c_callback = dispatcher.wrap_callback(python_callback, c_callback_type)
  ```
  """

  _callback: Optional[Callable[..., None]] = None
  _data_queue: queue.Queue[_AsyncResultPacket]
  _pipe_read_fd: int = _PIPE_NOT_INITIALIZED
  _pipe_write_fd: int = _PIPE_NOT_INITIALIZED
  _pipe_signaled: bool = False
  _state: _DispatcherState = _DispatcherState.NOT_STARTED
  _dispatcher_thread: Optional[threading.Thread] = None

  def __init__(self, converter: Callable[..., Any]):
    """Initializes the AsyncResultDispatcher.

    Args:
      converter: A function that takes the raw C-style arguments from the C
        library callback and converts them into Python-friendly types.
    """
    self._converter = converter
    self._data_queue = queue.Queue()
    self._lock = threading.Lock()

  def wrap_callback(
      self,
      python_callback: Optional[Callable[..., None]],
      c_callback_type: type[CCallbackType],
  ) -> CCallbackType:
    """Returns a ctypes callback function that can be passed to a C library.

    This function calls the converter function and schedules the user callback
    for execution with the converted data. This function can only be called
    once per instance.

    The C callback must take the user data, the status code and the error
    message as its first three arguments. The remaining arguments are passed
    to the converter function. The user data is ignored, since the returned
    callback is already bound to this dispatcher.

    Args:
      python_callback: The user-defined result callback for processing
        livestream data.
      c_callback_type: The ctypes callback function wrapper of the function you
        wish to create.

    Raises:
      RuntimeError: If a callback has already been assigned.
    """
    if python_callback is None:
      return c_callback_type()

    if self._callback is not None:
      raise RuntimeError("Callback already set.")
    self._callback = python_callback

    def c_callback(
        user_data: Optional[int],
        status_code: int,
        error_message: Optional[bytes],
        *args: Any,
    ) -> None:
      """The callback function that is invoked by the C library.

      Args:
        user_data: The user data that is passed to the C library callback.
        status_code: The status code that is passed to the C library callback.
        error_message: The error message that is passed to the C library
          callback, or None if the status code is OK.
        *args: Any additional arguments passed to the C library callback.
      """
      del user_data  # The callback is bound to this dispatcher.
      if status_code != 0:
        try:
          exception = mediapipe_c_utils.convert_to_exception(
              status_code, ctypes.c_char_p(error_message)
          )
          if exception is not None:
            self._put_packet(_ExceptionPacket(exception=exception))
        except Exception as e:  # pylint: disable=broad-except
          self._put_packet(_ExceptionPacket(exception=e))
        return

      try:
        py_args = self._converter(*args)
        self._put_packet(LiveStreamPacket(contents=py_args))
      except Exception as e:  # pylint: disable=broad-except
        self._put_packet(_ExceptionPacket(exception=e))

    return c_callback_type(c_callback)

  def _process_packet(self, packet: _AsyncResultPacket) -> None:
    """Processes a single packet from the data queue."""
    if isinstance(packet, _ExceptionPacket):
      logging.error("Error in async operation: %r", packet.exception)
    elif isinstance(packet, LiveStreamPacket) and self._callback:
      try:
        self._callback(*packet.contents)
      except Exception as e:  # pylint: disable=broad-except
        logging.exception("Error in callback: %r", e)

  def _close_pipes(self) -> None:
    """Closes OS pipe file descriptors if still open."""
    with self._lock:
      read_fd, self._pipe_read_fd = self._pipe_read_fd, _PIPE_NOT_INITIALIZED
    if read_fd != _PIPE_NOT_INITIALIZED:
      try:
        os.close(read_fd)
      except OSError:
        pass
    with self._lock:
      write_fd, self._pipe_write_fd = self._pipe_write_fd, _PIPE_NOT_INITIALIZED
      if write_fd != _PIPE_NOT_INITIALIZED:
        try:
          os.close(write_fd)
        except OSError:
          pass

  def _dispatcher_loop(self) -> None:
    """The main loop for the dispatcher thread."""
    try:
      while True:
        # Drain elements that are currently in the queue.
        while not self._data_queue.empty():
          packet = self._data_queue.get_nowait()
          self._process_packet(packet)

        with self._lock:
          if not self._data_queue.empty():
            continue
          if self._state == _DispatcherState.SHUTTING_DOWN:
            break
          read_fd = self._pipe_read_fd

        if read_fd == _PIPE_NOT_INITIALIZED:
          break

        try:
          # Block until the dispatcher is signaled again.
          os.read(read_fd, 1)
        except OSError:
          # Pipe might error during shutdown. Ignore.
          continue
        with self._lock:
          self._pipe_signaled = False
    finally:
      self._close_pipes()

  def _signal_locked(self) -> None:
    """Wakes up the dispatcher thread while holding `self._lock`.

    Signals are coalesced until the dispatcher thread consumed the previous
    one. The pipe therefore never holds more than one byte, and writing to it
    cannot block while `self._lock` is held, even if the user callback is slower
    than the C library.
    """
    if self._pipe_signaled or self._pipe_write_fd == _PIPE_NOT_INITIALIZED:
      return
    try:
      os.write(self._pipe_write_fd, b"\0")
    except OSError:
      # Pipe might error during shutdown. Ignore.
      return
    self._pipe_signaled = True

  def _put_packet(self, packet: _AsyncResultPacket) -> None:
    """Puts a data packet into the queue and signals the dispatcher thread."""
    with self._lock:
      if self._state == _DispatcherState.NOT_STARTED:
        self._start_locked()
      elif self._state == _DispatcherState.SHUTTING_DOWN:
        return

      self._data_queue.put(packet)
      self._signal_locked()

  def close(self) -> None:
    """Shuts down the dispatcher thread and cleans up resources.

    Waits until the dispatcher thread delivered all pending results, unless
    this is called by the user callback or while the interpreter shuts down.
    """
    with self._lock:
      self._state = _DispatcherState.SHUTTING_DOWN
      # Unblock the os.read() call so that the thread notices the new state.
      self._signal_locked()
      dispatcher_thread = self._dispatcher_thread

    # The dispatcher thread closes the pipes when it exits. It cannot join
    # itself, and daemon threads no longer run once the interpreter shuts down.
    if (
        dispatcher_thread is not None
        and dispatcher_thread is not threading.current_thread()
        and not mediapipe_c_utils.is_shutdown()
    ):
      dispatcher_thread.join()

  def _start_locked(self) -> None:
    """Starts the dispatcher thread while holding `self._lock`."""
    if self._callback is None:
      raise ValueError("Dispatcher should not be started without a callback.")
    self._state = _DispatcherState.RUNNING
    self._pipe_read_fd, self._pipe_write_fd = os.pipe()
    self._dispatcher_thread = threading.Thread(
        target=self._dispatcher_loop, daemon=True
    )
    self._dispatcher_thread.start()
