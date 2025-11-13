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

import ctypes
import threading
from typing import Any
from unittest import mock

from absl.testing import absltest

from mediapipe.tasks.python.core import mediapipe_c_utils
from mediapipe.tasks.python.core import serial_dispatcher


def _register_func(
    func_name, argtypes: list[Any] | None = None, return_type=ctypes.c_void_p
):
  """Creates the signature of a function to register in the dispatcher."""
  return mediapipe_c_utils.CFunction(
      func_name=func_name,
      argtypes=argtypes if argtypes else [],
      restype=return_type,
  )


class SerialDispatcherTest(absltest.TestCase):

  def test_delegates_to_registered_methods(self):
    mock_lib = mock.MagicMock()
    mock_lib.my_method = mock.MagicMock()
    signatures = [_register_func("my_method")]

    with serial_dispatcher.SerialDispatcher(mock_lib, signatures) as dispatcher:
      dispatcher.my_method()
    mock_lib.my_method.assert_called_once()

  def test_method_calls_are_serialized(self):
    mock_lib = mock.MagicMock()
    mock_long_running_func = mock_lib.long_running_func
    mock_queued_func = mock_lib.queued_func

    # Set up events to control the execution flow of long_running_func.
    long_running_func_has_started = threading.Event()
    long_running_func_may_complete = threading.Event()

    def side_effect_long_running_func():
      long_running_func_has_started.set()
      long_running_func_may_complete.wait()

    mock_long_running_func.side_effect = side_effect_long_running_func

    signatures = [
        _register_func("long_running_func"),
        _register_func("queued_func"),
    ]

    with serial_dispatcher.SerialDispatcher(mock_lib, signatures) as dispatcher:
      thread1 = threading.Thread(target=dispatcher.long_running_func)
      thread1.start()

      long_running_func_has_started.wait()

      thread2 = threading.Thread(target=dispatcher.queued_func)
      thread2.start()

      # Ensure that long_running_func blocks the call to queued_func.
      mock_long_running_func.assert_called_once()
      mock_queued_func.assert_not_called()

      # Allow long_running_func to complete, which allows queued_func to run.
      long_running_func_may_complete.set()
      thread2.join()
      mock_queued_func.assert_called_once()

  def test_method_calls_are_invoked_from_same_thread(self):
    num_threads = 5
    mock_lib = mock.MagicMock()

    thread_ids = set()

    def write_thread_id():
      thread_ids.add(threading.get_ident())

    mock_lib.write_thread_id = write_thread_id
    signatures = [_register_func("write_thread_id")]

    with serial_dispatcher.SerialDispatcher(mock_lib, signatures) as dispatcher:
      threads = []
      for _ in range(num_threads):
        thread = threading.Thread(target=dispatcher.write_thread_id)
        threads.append(thread)
        thread.start()

      for thread in threads:
        thread.join()

      self.assertLen(thread_ids, 1)

  def test_returns_value(self):
    mock_lib = mock.MagicMock()
    mock_lib.return_42.return_value = 42

    signatures = [_register_func("return_42", return_type=ctypes.c_int)]

    with serial_dispatcher.SerialDispatcher(mock_lib, signatures) as dispatcher:
      self.assertEqual(dispatcher.return_42(), 42)

  def test_raises_error(self):
    mock_lib = mock.MagicMock()
    mock_lib.error_func.side_effect = ValueError("Test Error")

    signatures = [_register_func("error_func")]

    with serial_dispatcher.SerialDispatcher(mock_lib, signatures) as dispatcher:
      with self.assertRaisesRegex(ValueError, "Test Error"):
        dispatcher.error_func()

  def test_continues_after_error(self):
    mock_lib = mock.MagicMock()
    mock_lib.error_func.side_effect = ValueError("Test Error")
    mock_lib.return_42.return_value = 42

    signatures = [
        _register_func("error_func"),
        _register_func("return_42", return_type=ctypes.c_int),
    ]

    with serial_dispatcher.SerialDispatcher(mock_lib, signatures) as dispatcher:
      try:
        dispatcher.error_func()
      except ValueError:
        pass

      # Ensure that we can still make calls after an exception.
      self.assertEqual(dispatcher.return_42(), 42)

  def test_calls_after_close_are_not_dispatched(self):
    mock_lib = mock.MagicMock()
    mock_lib.some_func.returns_42 = 42
    signatures = [_register_func("returns_42")]

    dispatcher = serial_dispatcher.SerialDispatcher(mock_lib, signatures)
    dispatcher.close()

    # The dispatcher returns a default value of None for all calls after its
    # closed.
    # TODO: b/456183832 - Return 0 once all APIs return MpStatus.
    self.assertIsNone(dispatcher.returns_42())


if __name__ == "__main__":
  absltest.main()
