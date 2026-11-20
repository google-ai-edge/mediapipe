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
import gc
import threading
from typing import Any
from unittest import mock
import weakref

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

  def test_gc_collect_open_task_in_reference_cycle_does_not_hang(self):
    closed_handles = []

    class _TaskLike:
      _lib: Any
      _handle: int | None

      def __init__(self, handle: int):
        mock_lib = mock.MagicMock(spec_set=["run_op", "close_op"])
        mock_lib.close_op.side_effect = closed_handles.append
        self._lib = serial_dispatcher.SerialDispatcher(
            mock_lib,
            [
                _register_func("run_op"),
                _register_func("close_op", argtypes=[ctypes.c_int]),
            ],
        )
        self._handle = handle
        self._lib.run_op()  # pyrefly: ignore[missing-attribute]

      def close(self):
        if not self._handle:
          return
        handle = self._handle
        self._handle = None
        try:
          self._lib.close_op(handle)
        finally:
          self._lib.close()

      def __del__(self):
        self.close()

    for i in range(1, 16):
      task = _TaskLike(i)
      cycle: list[Any] = [task]
      cycle.append(cycle)
      del task, cycle
      gc.collect()

    self.assertEqual(closed_handles, list(range(1, 16)))

  def test_gc_during_another_dispatcher_call_does_not_deadlock(self):
    closed_in_gc = threading.Event()

    class _CyclicTask:

      def __init__(self):
        mock_lib = mock.MagicMock(spec_set=["close_op"])
        mock_lib.close_op.side_effect = closed_in_gc.set
        self._lib: Any = serial_dispatcher.SerialDispatcher(
            mock_lib, [_register_func("close_op")]
        )
        self._open = True

      def close(self):
        if not self._open:
          return
        self._open = False
        try:
          self._lib.close_op()
        finally:
          self._lib.close()

      def __del__(self):
        self.close()

    cyclic_task = _CyclicTask()
    cycle: list[Any] = [cyclic_task]
    cycle.append(cycle)
    del cyclic_task, cycle

    other_lib = mock.MagicMock(spec_set=["trigger_gc"])
    other_lib.trigger_gc.side_effect = gc.collect
    with serial_dispatcher.SerialDispatcher(
        other_lib, [_register_func("trigger_gc")]
    ) as other_dispatcher:
      other_dispatcher.trigger_gc()  # pyrefly: ignore[missing-attribute]

    self.assertTrue(closed_in_gc.is_set())

  def test_raised_error_does_not_create_reference_cycle(self):
    mock_lib = mock.MagicMock(spec_set=["invalid_op", "MpErrorFree"])

    def invalid_op(error_msg):
      error_msg._obj.value = b"bad input"
      return 3

    mock_lib.invalid_op.side_effect = invalid_op
    signatures = [
        mediapipe_c_utils.CStatusFunction(
            func_name="invalid_op", core_argtypes=[]
        )
    ]

    class _Holder:

      def __init__(self):
        self.dispatcher: Any = serial_dispatcher.SerialDispatcher(
            mock_lib, signatures
        )

      def run(self):
        self.dispatcher.invalid_op()

    gc.disable()
    try:
      holder = _Holder()
      holder_ref = weakref.ref(holder)
      with self.assertRaisesRegex(ValueError, "bad input"):
        holder.run()
      holder.dispatcher.close()
      del holder
      self.assertIsNone(holder_ref())
    finally:
      gc.enable()

  def test_concurrent_call_and_close_never_deadlocks_after_sentinel(self):
    for _ in range(25):
      mock_lib = mock.MagicMock(spec_set=["work"])
      mock_lib.work.return_value = 7
      dispatcher: Any = serial_dispatcher.SerialDispatcher(
          mock_lib, [_register_func("work")]
      )
      barrier = threading.Barrier(2)

      def _caller(d=dispatcher, b=barrier):
        b.wait()
        for _ in range(10):
          d.work()

      caller_thread = threading.Thread(target=_caller)
      caller_thread.start()
      barrier.wait()
      dispatcher.close()
      caller_thread.join(timeout=5.0)
      self.assertFalse(caller_thread.is_alive())

  def test_concurrent_close_calls_wait_for_running_call(self):
    call_has_started = threading.Event()
    call_may_complete = threading.Event()
    self.addCleanup(call_may_complete.set)
    worker_threads = []

    def long_running_func():
      worker_threads.append(threading.current_thread())
      call_has_started.set()
      call_may_complete.wait()

    mock_lib = mock.MagicMock(spec_set=["long_running_func"])
    mock_lib.long_running_func.side_effect = long_running_func
    dispatcher: Any = serial_dispatcher.SerialDispatcher(
        mock_lib, [_register_func("long_running_func")]
    )
    caller = threading.Thread(target=dispatcher.long_running_func, daemon=True)
    caller.start()
    call_has_started.wait()

    closers = [
        threading.Thread(target=dispatcher.close, daemon=True),
        threading.Thread(target=dispatcher.close, daemon=True),
    ]
    for closer in closers:
      closer.start()
    for closer in closers:
      closer.join(timeout=0.1)
      self.assertTrue(closer.is_alive())

    call_may_complete.set()
    for closer in closers:
      closer.join(timeout=5.0)
      self.assertFalse(closer.is_alive())
    self.assertFalse(worker_threads[0].is_alive())

  def test_call_made_during_another_call_runs_on_worker_thread(self):
    thread_ids = []

    def outer_func():
      thread_ids.append(threading.get_ident())
      return dispatcher.inner_func() + 1

    def inner_func():
      thread_ids.append(threading.get_ident())
      return 41

    mock_lib = mock.MagicMock(spec_set=["outer_func", "inner_func"])
    mock_lib.outer_func.side_effect = outer_func
    mock_lib.inner_func.side_effect = inner_func
    dispatcher: Any = serial_dispatcher.SerialDispatcher(
        mock_lib,
        [
            _register_func("outer_func", return_type=ctypes.c_int),
            _register_func("inner_func", return_type=ctypes.c_int),
        ],
    )

    results = []
    caller = threading.Thread(
        target=lambda: results.append(dispatcher.outer_func()), daemon=True
    )
    caller.start()
    caller.join(timeout=5.0)

    self.assertEqual(results, [42])
    self.assertLen(thread_ids, 2)
    self.assertEqual(thread_ids[0], thread_ids[1])
    dispatcher.close()

  def test_call_arguments_are_released_when_call_returns(self):
    class _Argument:
      pass

    class _FakeLib:

      def __init__(self):
        self.consume = lambda argument: None

    fake_lib: Any = _FakeLib()
    dispatcher: Any = serial_dispatcher.SerialDispatcher(
        fake_lib, [_register_func("consume", argtypes=[ctypes.py_object])]
    )

    gc.disable()
    try:
      argument = _Argument()
      argument_ref = weakref.ref(argument)
      dispatcher.consume(argument)
      del argument
      self.assertIsNone(argument_ref())
    finally:
      gc.enable()
    dispatcher.close()

  def test_calls_run_on_daemon_thread(self):
    mock_lib = mock.MagicMock(spec_set=["current_thread"])
    mock_lib.current_thread.side_effect = threading.current_thread

    with serial_dispatcher.SerialDispatcher(
        mock_lib, [_register_func("current_thread")]
    ) as dispatcher:
      worker_thread = dispatcher.current_thread()  # pyrefly: ignore[missing-attribute]

    # A dispatcher that is never closed must not keep the interpreter alive.
    self.assertTrue(worker_thread.daemon)


if __name__ == "__main__":
  absltest.main()
