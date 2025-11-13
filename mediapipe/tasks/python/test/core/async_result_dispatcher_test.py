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
from unittest import mock

from absl.testing import absltest

from mediapipe.tasks.python.core import async_result_dispatcher
from mediapipe.tasks.python.core import mediapipe_c_bindings

_AsyncResultDispatcher = async_result_dispatcher.AsyncResultDispatcher
_MpStatus = mediapipe_c_bindings.MpStatus

_MP_STATUS_OK = _MpStatus.MP_OK.value
_C_CALLBACK_TYPE = ctypes.CFUNCTYPE(
    None, ctypes.c_int32, ctypes.c_char_p, ctypes.c_char_p
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
    c_types_callback(_MP_STATUS_OK, b"1", b"1")
    dispatcher.close()

    mock_callback.assert_called_once_with(1, 1)

  def test_exception_in_callback_does_not_block_dispatcher(self):
    mock_callback = mock.Mock()
    mock_callback.side_effect = RuntimeError("Callback error")
    dispatcher = _AsyncResultDispatcher(converter=_to_int_converter)
    c_types_callback = dispatcher.wrap_callback(
        mock_callback, _C_CALLBACK_TYPE
    )
    c_types_callback(_MpStatus.MP_OK.value, b"1", b"2")
    c_types_callback(_MpStatus.MP_OK.value, b"3", b"4")
    dispatcher.close()

    # Check that the callback was called twice, even though the calls raise
    # exceptions.
    mock_callback.assert_has_calls([mock.call(1, 2), mock.call(3, 4)])

  def test_close_flushes_queue(self):
    mock_callback = mock.Mock()

    dispatcher = _AsyncResultDispatcher(converter=_to_int_converter)
    c_callback = dispatcher.wrap_callback(mock_callback, _C_CALLBACK_TYPE)
    c_callback(_MpStatus.MP_OK.value, b"1", b"1")
    c_callback(_MpStatus.MP_OK.value, b"2", b"2")
    c_callback(_MpStatus.MP_OK.value, b"3", b"3")
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
    c_types_callback(_MpStatus.MP_OK.value, b"1", b"1")
    dispatcher.close()

    self.assertIsNotNone(callback_thread_id)
    self.assertNotEqual(main_thread_id, callback_thread_id)

  def test_ignores_calls_after_close(self):
    mock_callback = mock.Mock()
    dispatcher = _AsyncResultDispatcher(converter=_to_int_converter)
    c_callback = dispatcher.wrap_callback(mock_callback, _C_CALLBACK_TYPE)
    dispatcher.close()

    c_callback(_MpStatus.MP_OK.value, b"100", b"100")

    self.assertFalse(mock_callback.called)


if __name__ == "__main__":
  absltest.main()
