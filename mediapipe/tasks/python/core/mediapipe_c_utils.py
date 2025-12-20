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
import concurrent.futures
import ctypes
import dataclasses
import enum
import functools
import threading
from typing import Any, Callable, Optional, Sequence, Union


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

  def create_python_wrapper(
      self, lib: ctypes.CDLL, executor: concurrent.futures.Executor
  ) -> Callable[..., Any]:
    """Creates a callable wrapper that executes the C function on the executor.

    Args:
      lib: The ctypes shared library.
      executor: The executor to use for the C function.

    Returns:
      A callable that executes the C function on the executor.
    """
    c_func = getattr(lib, self.func_name)
    c_func.argtypes = self.argtypes
    c_func.restype = self.restype

    @functools.wraps(c_func)
    def dispatcher_wrapper(*args, **kwargs) -> Any:
      future = executor.submit(c_func, *args, **kwargs)
      return future.result()

    return dispatcher_wrapper


@dataclasses.dataclass(frozen=True)
class CStatusFunction(CFunction):
  """Stores the ctypes signature for a status-returning C function.

  The functions is required to take an error message argument as the last
  argument and return an `MpStatus' integer code.
  """

  def __init__(self, func_name: str, core_argtypes: Sequence[Any]):
    """Initializes the CStatusFunction.

    Args:
      func_name: The name of the C function.
      core_argtypes: The ctypes types for the function arguments excluding the
        error message argument.
    """
    full_argtypes = list(core_argtypes) + [ctypes.POINTER(ctypes.c_char_p)]
    super().__init__(func_name, full_argtypes, restype=ctypes.c_int)

  def create_python_wrapper(
      self, lib: ctypes.CDLL, executor: concurrent.futures.Executor
  ) -> Callable[..., None]:
    """Creates a callable wrapper that executes the C function on the executor.

    This function raises a Python exception based on the MpStatus code returned
    by the C API function and the error message argument.

    Args:
      lib: The ctypes shared library.
      executor: The executor to use for the C function.

    Returns:
      A callable that executes the C function on the executor.
    """
    c_func = getattr(lib, self.func_name)
    c_func.argtypes = self.argtypes
    c_func.restype = self.restype
    error_msg_index = len(self.argtypes)

    @functools.wraps(c_func)
    def dispatcher_wrapper(*args, **kwargs) -> None:
      if is_shutdown():
        return

      def dispatch_and_free():
        # Add the error message argument to the list of arguments.
        error_msg = ctypes.c_char_p()
        call_args = list(args)
        call_args.insert(error_msg_index, ctypes.byref(error_msg))

        status = c_func(*call_args, **kwargs)
        try:
          handle_status(status, error_msg)
        finally:
          if error_msg.value is not None:
            lib.MpErrorFree(error_msg)

      executor.submit(dispatch_and_free).result()

    return dispatcher_wrapper


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


def convert_to_exception(
    status: int, error_message: Optional[ctypes.c_char_p] = None
) -> Union[Exception, None]:
  """Returns an exception based on the MpStatus code, or None if MP_OK."""
  error_str = (
      error_message.value.decode('utf-8')
      if error_message and error_message.value is not None
      else None
  )

  if status == MpStatus.MP_OK:
    return None
  elif status == MpStatus.MP_CANCELLED:
    return TimeoutError(error_str or 'Cancelled')
  elif status == MpStatus.MP_UNKNOWN:
    return RuntimeError(error_str or 'Unknown error')
  elif status == MpStatus.MP_INVALID_ARGUMENT:
    return ValueError(error_str or 'Invalid argument')
  elif status == MpStatus.MP_DEADLINE_EXCEEDED:
    return TimeoutError(error_str or 'Deadline exceeded')
  elif status == MpStatus.MP_NOT_FOUND:
    return FileNotFoundError(error_str or 'Not found')
  elif status == MpStatus.MP_ALREADY_EXISTS:
    return FileExistsError(error_str or 'Already exists')
  elif status == MpStatus.MP_PERMISSION_DENIED:
    return PermissionError(error_str or 'Permission denied')
  elif status == MpStatus.MP_RESOURCE_EXHAUSTED:
    return RuntimeError(error_str or 'Resource exhausted')
  elif status == MpStatus.MP_FAILED_PRECONDITION:
    return RuntimeError(error_str or 'Failed precondition')
  elif status == MpStatus.MP_ABORTED:
    return RuntimeError(error_str or 'Aborted')
  elif status == MpStatus.MP_OUT_OF_RANGE:
    return IndexError(error_str or 'Out of range')
  elif status == MpStatus.MP_UNIMPLEMENTED:
    return NotImplementedError(error_str or 'Unimplemented')
  elif status == MpStatus.MP_INTERNAL:
    return RuntimeError(error_str or 'Internal error')
  elif status == MpStatus.MP_UNAVAILABLE:
    return RuntimeError(error_str or 'Unavailable')
  elif status == MpStatus.MP_DATA_LOSS:
    return RuntimeError(error_str or 'Data loss')
  elif status == MpStatus.MP_UNAUTHENTICATED:
    return PermissionError(error_str or 'Unauthenticated')
  else:
    return RuntimeError(error_str or f'Unexpected status: {status}')


def handle_status(
    status: int, error_message: Optional[ctypes.c_char_p] = None
) -> None:
  """Checks the MpStatus and raises an error if not kMpOk.

  Args:
    status: The MpStatus code to check.
    error_message: The error message to use if an exception is raised. Uses a
      default error message if None.

  Raises:
    An error if the MpStatus is not kMpOk.
  """
  exception = convert_to_exception(status, error_message)
  if exception:
    raise exception
