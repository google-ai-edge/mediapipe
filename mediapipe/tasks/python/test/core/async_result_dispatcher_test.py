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
import os
import queue
import threading
from unittest import mock

from absl.testing import absltest

from mediapipe.tasks.python.core import async_result_dispatcher
from mediapipe.tasks.python.core import mediapipe_c_utils

_AsyncResultDispatcher = async_result_dispatcher.AsyncResultDispatcher
_MpStatus = mediapipe_c_utils.MpStatus

_MP_STATUS_OK = _MpStatus.MP_OK.value
_C_CALLBACK_TYPE = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,  # User data
    ctypes.c_int32,  # Status code
    ctypes.c_char_p,  # Error message
    ctypes.c_char_p,
    ctypes.c_char_p,
)


def _to_int_converter(first: str, second: str) -> tuple[int, int]:
  return int(first), int(second)


class AsyncResultDispatcherTest(absltest.TestCase):

  def test_throws_error_if_callback_already_set(self):
    dispatcher = _AsyncResultDispatcher(converter=_to_int_converter)
    mock_callback = mock.Mock()
    dispatcher.wrap_callback(mock_callback, _C_CALLBACK_TYPE)

    with self.assertRaisesRegex(RuntimeError, "Callback already set."):
      dispatcher.wrap_callback(mock_callback, _C_CALLBACK_TYPE)

  def test_callback_invoked_with_converted_data(self):
    mock_callback = mock.Mock()
    dispatcher = _AsyncResultDispatcher(converter=_to_int_converter)
    c_types_callback = dispatcher.wrap_callback(
        mock_callback, _C_CALLBACK_TYPE
    )
    c_types_callback(None, _MP_STATUS_OK, None, b"1", b"1")
    dispatcher.close()

    mock_callback.assert_called_once_with(1, 1)

  def test_exception_in_callback_does_not_block_dispatcher(self):
    mock_callback = mock.Mock()
    mock_callback.side_effect = RuntimeError("Callback error")
    dispatcher = _AsyncResultDispatcher(converter=_to_int_converter)
    c_types_callback = dispatcher.wrap_callback(
        mock_callback, _C_CALLBACK_TYPE
    )
    c_types_callback(None, _MpStatus.MP_OK.value, None, b"1", b"2")
    c_types_callback(None, _MpStatus.MP_OK.value, None, b"3", b"4")
    dispatcher.close()

    # Check that the callback was called twice, even though the calls raise
    # exceptions.
    mock_callback.assert_has_calls([mock.call(1, 2), mock.call(3, 4)])

  def test_error_is_logged_with_message_and_does_not_invoke_callback(self):
    mock_callback = mock.Mock()
    dispatcher = _AsyncResultDispatcher(converter=_to_int_converter)
    c_types_callback = dispatcher.wrap_callback(mock_callback, _C_CALLBACK_TYPE)

    with self.assertLogs(level="ERROR") as logs:
      c_types_callback(
          None,
          _MpStatus.MP_INVALID_ARGUMENT.value,
          b"The image is invalid.",
          None,
          None,
      )
      dispatcher.close()

    self.assertLen(logs.output, 1)
    self.assertIn("The image is invalid.", logs.output[0])
    mock_callback.assert_not_called()

  def test_error_is_logged_when_message_is_none_or_empty(self):
    mock_callback = mock.Mock()
    dispatcher = _AsyncResultDispatcher(converter=_to_int_converter)
    c_types_callback = dispatcher.wrap_callback(mock_callback, _C_CALLBACK_TYPE)

    with self.assertLogs(level="ERROR") as logs:
      c_types_callback(
          None, _MpStatus.MP_INVALID_ARGUMENT.value, None, None, None
      )
      c_types_callback(
          None, _MpStatus.MP_INVALID_ARGUMENT.value, b"", None, None
      )
      dispatcher.close()

    self.assertLen(logs.output, 2)
    self.assertIn("Invalid argument", logs.output[0])
    self.assertIn("Invalid argument", logs.output[1])
    mock_callback.assert_not_called()

  def test_error_with_invalid_utf8_bytes_is_logged_with_replacement(self):
    mock_callback = mock.Mock()
    dispatcher = _AsyncResultDispatcher(converter=_to_int_converter)
    c_types_callback = dispatcher.wrap_callback(mock_callback, _C_CALLBACK_TYPE)

    with self.assertLogs(level="ERROR") as logs:
      c_types_callback(
          None,
          _MpStatus.MP_NOT_FOUND.value,
          b"Invalid path: \xff\xfe",
          None,
          None,
      )
      dispatcher.close()

    self.assertLen(logs.output, 1)
    self.assertIn("Invalid path: \ufffd\ufffd", logs.output[0])
    mock_callback.assert_not_called()

  def test_user_data_is_ignored(self):
    mock_callback = mock.Mock()
    dispatcher = _AsyncResultDispatcher(converter=_to_int_converter)
    c_types_callback = dispatcher.wrap_callback(mock_callback, _C_CALLBACK_TYPE)
    c_types_callback(1234, _MP_STATUS_OK, None, b"1", b"2")
    dispatcher.close()

    mock_callback.assert_called_once_with(1, 2)

  def test_close_flushes_queue(self):
    mock_callback = mock.Mock()

    dispatcher = _AsyncResultDispatcher(converter=_to_int_converter)
    c_callback = dispatcher.wrap_callback(mock_callback, _C_CALLBACK_TYPE)
    c_callback(None, _MpStatus.MP_OK.value, None, b"1", b"1")
    c_callback(None, _MpStatus.MP_OK.value, None, b"2", b"2")
    c_callback(None, _MpStatus.MP_OK.value, None, b"3", b"3")
    dispatcher.close()

    self.assertEqual(mock_callback.call_count, 3)

  def test_calls_callbck_in_separate_thread(self):
    main_thread_id = threading.get_ident()
    callback_thread_id = None

    mock_callback = mock.Mock()

    def side_effect(*args, **kwargs):
      del args, kwargs
      nonlocal callback_thread_id
      callback_thread_id = threading.get_ident()

    mock_callback.side_effect = side_effect

    dispatcher = _AsyncResultDispatcher(converter=_to_int_converter)
    c_types_callback = dispatcher.wrap_callback(
        mock_callback, _C_CALLBACK_TYPE
    )
    c_types_callback(None, _MpStatus.MP_OK.value, None, b"1", b"1")
    dispatcher.close()

    self.assertIsNotNone(callback_thread_id)
    self.assertNotEqual(main_thread_id, callback_thread_id)

  def test_ignores_calls_after_close(self):
    mock_callback = mock.Mock()
    dispatcher = _AsyncResultDispatcher(converter=_to_int_converter)
    c_callback = dispatcher.wrap_callback(mock_callback, _C_CALLBACK_TYPE)
    dispatcher.close()

    c_callback(None, _MpStatus.MP_OK.value, None, b"100", b"100")

    self.assertFalse(mock_callback.called)
    self.assertIsNone(dispatcher._dispatcher_thread)

  def test_concurrent_callbacks_and_close_do_not_corrupt_pipes(self):
    for _ in range(25):
      dispatcher = _AsyncResultDispatcher(converter=_to_int_converter)
      c_callback = dispatcher.wrap_callback(
          lambda a, b: None, _C_CALLBACK_TYPE
      )
      c_callback(None, _MP_STATUS_OK, None, b"1", b"2")

      barrier = threading.Barrier(2)

      def _spam_callbacks(cb=c_callback, b=barrier):
        b.wait()
        for _ in range(20):
          cb(None, _MP_STATUS_OK, None, b"3", b"4")

      writer = threading.Thread(target=_spam_callbacks)
      writer.start()
      barrier.wait()
      dispatcher.close()
      writer.join(timeout=5.0)
      self.assertFalse(writer.is_alive())

  def test_delivers_each_result_before_close(self):
    received = queue.Queue()
    dispatcher = _AsyncResultDispatcher(converter=_to_int_converter)
    self.addCleanup(dispatcher.close)
    c_callback = dispatcher.wrap_callback(
        lambda first, second: received.put((first, second)), _C_CALLBACK_TYPE
    )

    # The dispatcher thread becomes idle between results, so each result has to
    # wake it up again.
    for i in range(100):
      c_callback(None, _MP_STATUS_OK, None, str(i).encode(), b"0")
      self.assertEqual(received.get(timeout=10.0), (i, 0))

  def test_delivers_results_that_arrive_during_slow_callback(self):
    # More results than fit into an OS pipe buffer on any supported platform.
    num_results = 100000
    first_callback_started = threading.Event()
    first_callback_may_return = threading.Event()
    all_results_received = threading.Event()
    received = []

    def slow_first_callback(first: int, second: int):
      del second
      if not received:
        first_callback_started.set()
        first_callback_may_return.wait()
      received.append(first)
      if len(received) == num_results:
        all_results_received.set()

    dispatcher = _AsyncResultDispatcher(converter=_to_int_converter)
    self.addCleanup(dispatcher.close)
    self.addCleanup(first_callback_may_return.set)
    c_callback = dispatcher.wrap_callback(slow_first_callback, _C_CALLBACK_TYPE)

    c_callback(None, _MP_STATUS_OK, None, b"0", b"0")
    first_callback_started.wait()
    for i in range(1, num_results):
      c_callback(None, _MP_STATUS_OK, None, str(i).encode(), b"0")
    first_callback_may_return.set()

    self.assertTrue(all_results_received.wait(timeout=60.0))
    self.assertEqual(received, list(range(num_results)))

  def test_close_from_callback_does_not_wait_for_itself(self):
    callback_returned = threading.Event()
    dispatcher = _AsyncResultDispatcher(converter=_to_int_converter)

    def close_dispatcher(first: int, second: int):
      del first, second
      dispatcher.close()
      callback_returned.set()

    c_callback = dispatcher.wrap_callback(close_dispatcher, _C_CALLBACK_TYPE)
    c_callback(None, _MP_STATUS_OK, None, b"1", b"2")
    dispatcher.close()

    self.assertTrue(callback_returned.is_set())

  def test_concurrent_close_calls_wait_for_running_callback(self):
    callback_started = threading.Event()
    callback_may_return = threading.Event()
    self.addCleanup(callback_may_return.set)

    def blocking_callback(first: int, second: int):
      del first, second
      callback_started.set()
      callback_may_return.wait()

    dispatcher = _AsyncResultDispatcher(converter=_to_int_converter)
    c_callback = dispatcher.wrap_callback(blocking_callback, _C_CALLBACK_TYPE)
    c_callback(None, _MP_STATUS_OK, None, b"1", b"2")
    callback_started.wait()

    closers = [
        threading.Thread(target=dispatcher.close, daemon=True),
        threading.Thread(target=dispatcher.close, daemon=True),
    ]
    for closer in closers:
      closer.start()
    for closer in closers:
      closer.join(timeout=0.1)
      self.assertTrue(closer.is_alive())

    callback_may_return.set()
    for closer in closers:
      closer.join(timeout=5.0)
      self.assertFalse(closer.is_alive())

  def test_close_does_not_wait_for_callback_during_interpreter_shutdown(self):
    callback_started = threading.Event()
    callback_may_return = threading.Event()
    self.addCleanup(callback_may_return.set)

    def blocking_callback(first: int, second: int):
      del first, second
      callback_started.set()
      callback_may_return.wait()

    dispatcher = _AsyncResultDispatcher(converter=_to_int_converter)
    c_callback = dispatcher.wrap_callback(blocking_callback, _C_CALLBACK_TYPE)
    c_callback(None, _MP_STATUS_OK, None, b"1", b"2")
    callback_started.wait()

    with mock.patch.object(
        mediapipe_c_utils, "is_shutdown", return_value=True, autospec=True
    ):
      closer = threading.Thread(target=dispatcher.close, daemon=True)
      closer.start()
      closer.join(timeout=5.0)

    self.assertFalse(closer.is_alive())

  def test_close_closes_pipe_file_descriptors(self):
    pipe_fds = []
    create_pipe = os.pipe

    def create_and_record_pipe():
      fds = create_pipe()
      pipe_fds.extend(fds)
      return fds

    dispatcher = _AsyncResultDispatcher(converter=_to_int_converter)
    c_callback = dispatcher.wrap_callback(
        lambda first, second: None, _C_CALLBACK_TYPE
    )
    with mock.patch.object(
        os, "pipe", side_effect=create_and_record_pipe, autospec=True
    ):
      c_callback(None, _MP_STATUS_OK, None, b"1", b"2")
    dispatcher.close()

    self.assertLen(pipe_fds, 2)
    for pipe_fd in pipe_fds:
      with self.assertRaisesRegex(OSError, "Bad file descriptor"):
        os.fstat(pipe_fd)


if __name__ == "__main__":
  absltest.main()
