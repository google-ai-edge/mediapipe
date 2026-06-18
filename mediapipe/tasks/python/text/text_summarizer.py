# Copyright 2026 The MediaPipe Authors.
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
"""MediaPipe text summarizer task."""

import ctypes
import dataclasses
import enum
from typing import Callable, Optional, cast

from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import base_options_c as base_options_c_module
from mediapipe.tasks.python.core import mediapipe_c_bindings
from mediapipe.tasks.python.core import mediapipe_c_utils
from mediapipe.tasks.python.core import serial_dispatcher


class Mode(enum.IntEnum):
  """The mode of the text summarizer."""
  TLDR = 0
  KEYPOINTS = 1


class _MpTextSummarizerOptionsC(ctypes.Structure):
  _fields_ = [
      ("base_options", base_options_c_module.MpBaseOptionsC),
      ("mode", ctypes.c_int),
      ("max_num_tokens", ctypes.c_int),
      ("cache_dir", ctypes.c_char_p),
  ]


@dataclasses.dataclass
class TextSummarizerOptions:
  """Options for the text summarizer task.

  Attributes:
    base_options: Base options for the text summarizer task.
    mode: The mode of the text summarizer task.
    max_num_tokens: The maximum number of tokens for summarization tasks. If
      set, the summarization will be truncated if the input and output exceed
      this value. If not set, then the default max_num_tokens is roughly 4000
      tokens due to the model's capacity.
  """

  base_options: base_options_module.BaseOptions
  mode: Mode = Mode.KEYPOINTS
  max_num_tokens: Optional[int] = None

  def to_ctypes(self) -> _MpTextSummarizerOptionsC:
    """Generates a ctypes TextSummarizerOptionsC."""
    base_options_c = self.base_options.to_ctypes()

    return _MpTextSummarizerOptionsC(
        base_options=base_options_c,
        mode=self.mode.value,
        max_num_tokens=(
            self.max_num_tokens if self.max_num_tokens is not None else 0
        ),
        cache_dir=None,
    )


class _MpTextSummarizerResultC(ctypes.Structure):
  _fields_ = [
      ("summary", ctypes.c_char_p),
  ]


class _MpTextSummarizerStreamResultC(ctypes.Structure):
  _fields_ = [
      ("chunk", ctypes.c_char_p),
      ("done", ctypes.c_bool),
  ]


@dataclasses.dataclass
class TextSummarizerResult:
  """Result of the text summarizer task.

  Attributes:
    summary: The generated summary text.
    done: Whether the summarization is done.
  """

  summary: Optional[str]
  done: bool


_SummarizerCallback = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.POINTER(_MpTextSummarizerStreamResultC),
    ctypes.c_char_p,
)


_CTYPES_SIGNATURES = (
    mediapipe_c_utils.CStatusFunction(
        "MpTextSummarizerCreate",
        [
            ctypes.POINTER(_MpTextSummarizerOptionsC),
            ctypes.POINTER(ctypes.c_void_p),
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        "MpTextSummarizerSummarize",
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(_MpTextSummarizerResultC),
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        "MpTextSummarizerSummarizeStreaming",
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            _SummarizerCallback,
            ctypes.c_void_p,
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        "MpTextSummarizerClose",
        [
            ctypes.c_void_p,
        ],
    ),
    mediapipe_c_utils.CFunction(
        "MpTextSummarizerCloseResult",
        [ctypes.POINTER(_MpTextSummarizerResultC)],
        None,
    ),
)


class TextSummarizer:
  """Performs summarization on text."""

  _lib: serial_dispatcher.SerialDispatcher
  _summarizer_handle: ctypes.c_void_p

  def __init__(
      self, lib: serial_dispatcher.SerialDispatcher, handle: ctypes.c_void_p
  ):
    self._lib = lib
    self._summarizer_handle = handle
    self._async_result_callback: Optional[
        Callable[[Optional[TextSummarizerResult], Optional[str]], None]
    ] = None

    def callback_wrapper(
        unused_user_data: int,
        result_ptr: ctypes.POINTER(_MpTextSummarizerStreamResultC),
        error_msg_ptr: bytes,
    ) -> None:
      if not self._async_result_callback:
        return
      is_done = False
      if error_msg_ptr:
        error_msg = error_msg_ptr.decode("utf-8")
        self._async_result_callback(None, error_msg)
        is_done = True
      elif result_ptr:
        c_result = cast(_MpTextSummarizerStreamResultC, result_ptr.contents)
        python_result = TextSummarizerResult(
            summary=c_result.chunk.decode("utf-8") if c_result.chunk else None,
            done=c_result.done,
        )
        self._async_result_callback(python_result, None)
        if c_result.done:
          is_done = True
      else:
        # Should not happen
        self._async_result_callback(None, "Unknown error")
        is_done = True

      if is_done:
        self._async_result_callback = None

    self._c_callback = _SummarizerCallback(callback_wrapper)

  @classmethod
  def create_from_model_path(cls, model_path: str) -> "TextSummarizer":
    """Creates a `TextSummarizer` object from model path.

    Args:
      model_path: Path to the model.

    Returns:
      `TextSummarizer` object that's created from the model file and the
      default `TextSummarizerOptions`.

    Raises:
      ValueError: If failed to create `TextSummarizer` object from the provided
        file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = base_options_module.BaseOptions(model_asset_path=model_path)
    options = TextSummarizerOptions(base_options=base_options)
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(
      cls, options: TextSummarizerOptions
  ) -> "TextSummarizer":
    """Creates the `TextSummarizer` object from text summarizer options.

    Args:
      options: Options for the text summarizer task.

    Returns:
      `TextSummarizer` object that's created from `options`.

    Raises:
      ValueError: If failed to create `TextSummarizer` object from
        `TextSummarizerOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """
    lib = mediapipe_c_bindings.load_shared_library(_CTYPES_SIGNATURES)

    ctypes_options = options.to_ctypes()

    summarizer_handle = ctypes.c_void_p()
    lib.MpTextSummarizerCreate(
        ctypes.byref(ctypes_options),
        ctypes.byref(summarizer_handle),
    )
    return TextSummarizer(lib=lib, handle=summarizer_handle)

  def summarize(self, text: str) -> TextSummarizerResult:
    """Performs summarization on the input `text`.

    Args:
      text: The input text.

    Returns:
      A `TextSummarizerResult` object that contains the summary.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If text summarization failed to run.
    """
    ctypes_result = _MpTextSummarizerResultC()

    self._lib.MpTextSummarizerSummarize(
        self._summarizer_handle,
        text.encode("utf-8"),
        ctypes.byref(ctypes_result),
    )
    python_result = TextSummarizerResult(
        summary=ctypes_result.summary.decode("utf-8")
        if ctypes_result.summary
        else None,
        done=True,
    )
    self._lib.MpTextSummarizerCloseResult(ctypes.byref(ctypes_result))
    return python_result

  def summarize_async(
      self,
      text: str,
      result_callback: Callable[
          [Optional[TextSummarizerResult], Optional[str]], None
      ],
  ) -> None:
    """Performs summarization on the input `text` asynchronously.

    Args:
      text: The input text.
      result_callback: A callback function that will be called with the
        TextSummarizerResult and an error message (if any). The callback
        signature is `(result: Optional[TextSummarizerResult], error:
        Optional[str]) -> None`.

    Raises:
      ValueError: If any of the input arguments is invalid, or if
        summarize_async is called before the previous call has finished.
      RuntimeError: If text summarization failed to run.
    """
    if self._async_result_callback:
      raise ValueError(
          "summarize_async is already running. Concurrent calls are not"
          " supported."
      )
    self._async_result_callback = result_callback
    self._lib.MpTextSummarizerSummarizeStreaming(
        self._summarizer_handle,
        text.encode("utf-8"),
        self._c_callback,
        None,
    )

  def close(self):
    """Shuts down the MediaPipe task instance."""
    if self._summarizer_handle:
      self._lib.MpTextSummarizerClose(self._summarizer_handle)
      self._summarizer_handle = None
      self._async_result_callback = None
      self._lib.close()

  def __enter__(self):
    """Returns `self` upon entering the runtime context."""
    return self

  def __exit__(self, unused_exc_type, unused_exc_value, unused_traceback):
    """Shuts down the MediaPipe task instance on exit of the context manager."""
    self.close()

  def __del__(self):
    self.close()
