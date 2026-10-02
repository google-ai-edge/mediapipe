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
from absl.testing import parameterized

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


class SerialDispatcherTest(parameterized.TestCase):

  def test_delegates_to_registered_methods(self):
    mock_lib = mock.MagicMock(spec_set=["my_method"])
    signatures = [_register_func("my_method")]

    with serial_dispatcher.SerialDispatcher(mock_lib, signatures) as dispatcher:
      dispatcher.my_method()
    mock_lib.my_method.assert_called_once()

  def test_method_calls_are_serialized(self):
    mock_lib = mock.MagicMock(spec_set=["long_running_func", "queued_func"])
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
    mock_lib = mock.MagicMock(spec_set=["write_thread_id"])

    thread_ids = set()

    def write_thread_id():
      thread_ids.add(threading.get_ident())

    mock_lib.write_thread_id.side_effect = write_thread_id
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
    mock_lib = mock.MagicMock(spec_set=["return_42"])
    mock_lib.return_42.return_value = 42

    signatures = [_register_func("return_42", return_type=ctypes.c_int)]

    with serial_dispatcher.SerialDispatcher(mock_lib, signatures) as dispatcher:
      self.assertEqual(dispatcher.return_42(), 42)

  def test_raises_error(self):
    mock_lib = mock.MagicMock(spec_set=["error_func"])
    mock_lib.error_func.side_effect = ValueError("Test Error")

    signatures = [_register_func("error_func")]

    with serial_dispatcher.SerialDispatcher(mock_lib, signatures) as dispatcher:
      with self.assertRaisesRegex(ValueError, "Test Error"):
        dispatcher.error_func()

  def test_continues_after_error(self):
    mock_lib = mock.MagicMock(spec_set=["error_func", "return_42"])
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
    mock_lib = mock.MagicMock(spec_set=["returns_42"])
    mock_lib.returns_42.return_value = 42
    signatures = [_register_func("returns_42")]

    dispatcher = serial_dispatcher.SerialDispatcher(mock_lib, signatures)
    dispatcher.close()

    # The dispatcher returns a default value of None for all calls after its
    # closed.
    self.assertIsNone(dispatcher.returns_42())  # pyrefly: ignore[missing-attribute]

  def test_calls_status_functions_with_error_argument(self):
    mock_lib = mock.MagicMock(spec_set=["status_method"])
    test_arg = 123

    def status_method(arg, error_msg) -> int:
      self.assertEqual(arg, test_arg)
      self.assertIsInstance(error_msg._obj, ctypes.c_char_p)
      return 0

    mock_lib.status_method.side_effect = status_method

    func_signatures = [
        mediapipe_c_utils.CStatusFunction(
            func_name="status_method",
            core_argtypes=[ctypes.c_int],
        )
    ]
    with serial_dispatcher.SerialDispatcher(
        mock_lib, func_signatures
    ) as dispatcher:
      dispatcher.status_method(test_arg)

    mock_lib.status_method.assert_called_once()

  @parameterized.named_parameters(
      ("ascii_error", "Test Error"),
      ("utf8_error", "⚠️"),
  )
  def test_uses_error_message_for_status_functions(self, error_message):
    mock_lib = mock.MagicMock(spec_set=["invalid_op", "MpErrorFree"])

    def invalid_op(error_msg):
      error_msg._obj.value = error_message.encode("utf-8")
      return 13

    mock_lib.invalid_op.side_effect = invalid_op

    func_signatures = [
        mediapipe_c_utils.CStatusFunction(
            func_name="invalid_op",
            core_argtypes=[],
        )
    ]
    dispatcher = serial_dispatcher.SerialDispatcher(mock_lib, func_signatures)

    with self.assertRaisesRegex(RuntimeError, error_message):
      dispatcher.invalid_op()  # pyrefly: ignore[missing-attribute]

    dispatcher.close()

  def test_task_close_function_runs_once_when_called_concurrently(self):
    mock_lib = mock.MagicMock(spec_set=["MpFooClose"])
    num_threads = 8
    start = threading.Barrier(num_threads)
    signatures = [_register_func("MpFooClose", [ctypes.c_void_p])]
    dispatcher = serial_dispatcher.SerialDispatcher(mock_lib, signatures)

    def close():
      start.wait()
      dispatcher.MpFooClose(1)

    threads = [threading.Thread(target=close) for _ in range(num_threads)]
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()

    # Freeing the native task twice would be a double free.
    mock_lib.MpFooClose.assert_called_once()
    dispatcher.close()

  def test_calls_queued_behind_task_close_function_are_not_dispatched(self):
    mock_lib = mock.MagicMock(spec_set=["MpFooClose", "MpFooUse"])
    use_started = threading.Event()
    use_may_complete = threading.Event()

    def use(*unused_args):
      use_started.set()
      use_may_complete.wait()

    mock_lib.MpFooUse.side_effect = use
    signatures = [
        _register_func("MpFooClose", [ctypes.c_void_p]),
        _register_func("MpFooUse", [ctypes.c_void_p]),
    ]
    dispatcher = serial_dispatcher.SerialDispatcher(mock_lib, signatures)

    in_flight = threading.Thread(target=dispatcher.MpFooUse, args=(1,))
    in_flight.start()
    use_started.wait()
    # Both of these wait behind the in-flight call. `MpFooClose` frees the
    # handle, so `MpFooUse` must not reach the library after it.
    closer = threading.Thread(target=dispatcher.MpFooClose, args=(1,))
    closer.start()
    late_user = threading.Thread(target=dispatcher.MpFooUse, args=(1,))
    late_user.start()
    use_may_complete.set()
    for thread in (in_flight, closer, late_user):
      thread.join()

    mock_lib.MpFooClose.assert_called_once()
    # The in-flight call, plus the late call only if it ran before the close.
    self.assertLessEqual(mock_lib.MpFooUse.call_count, 2)
    mock_lib.MpFooUse.reset_mock()
    self.assertIsNone(dispatcher.MpFooUse(1))
    mock_lib.MpFooUse.assert_not_called()
    dispatcher.close()

  def test_close_after_task_close_function_shuts_down_executor(self):
    mock_lib = mock.MagicMock(spec_set=["MpFooClose"])
    signatures = [_register_func("MpFooClose", [ctypes.c_void_p])]
    dispatcher = serial_dispatcher.SerialDispatcher(mock_lib, signatures)

    dispatcher.MpFooClose(1)
    dispatcher.close()

    with self.assertRaises(RuntimeError):
      dispatcher._executor.submit(lambda: None)


if __name__ == "__main__":
  absltest.main()
