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
"""MediaPipe text proofreader task."""

import ctypes
import dataclasses
import enum
from typing import Callable, List, Optional, Union, cast

from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import base_options_c as base_options_c_module
from mediapipe.tasks.python.core import mediapipe_c_bindings
from mediapipe.tasks.python.core import mediapipe_c_utils
from mediapipe.tasks.python.core import serial_dispatcher


class _MpTextProofreaderOptionsC(ctypes.Structure):
  _fields_ = [
      ("base_options", base_options_c_module.MpBaseOptionsC),
      ("max_num_tokens", ctypes.c_int),
      ("cache_dir", ctypes.c_char_p),
  ]


@dataclasses.dataclass
class TextProofreaderOptions:
  """Options for the text proofreader task.

  Attributes:
    base_options: Base options for the text proofreader task.
    max_num_tokens: The maximum number of tokens for proofreader tasks. If set,
      the proofread output text will be truncated if the input and output exceed
      this value. If not set, then the default max_num_tokens is roughly 8k
      tokens due to the model's capacity.
  """

  base_options: base_options_module.BaseOptions
  max_num_tokens: Optional[int] = None

  def to_ctypes(self) -> _MpTextProofreaderOptionsC:
    """Generates a ctypes TextProofreaderOptionsC."""
    base_options_c = self.base_options.to_ctypes()

    return _MpTextProofreaderOptionsC(
        base_options=base_options_c,
        max_num_tokens=(
            self.max_num_tokens if self.max_num_tokens is not None else 0
        ),
        cache_dir=None,
    )


class CorrectionType(enum.Enum):
  """The type of correction."""

  SAME = 0
  INSERTION = 1
  DELETION = 2


class _MpCorrectionC(ctypes.Structure):
  _fields_ = [
      ("type", ctypes.c_int),
      ("text", ctypes.c_char_p),
  ]


class _MpTextProofreaderResultC(ctypes.Structure):
  _fields_ = [
      ("proofread_text", ctypes.c_char_p),
      ("corrections_count", ctypes.c_int),
      ("corrections", ctypes.POINTER(_MpCorrectionC)),
  ]


class _MpTextProofreaderStreamResultC(ctypes.Structure):
  _fields_ = [
      ("chunk", ctypes.c_char_p),
      ("corrections_count", ctypes.c_int),
      ("corrections", ctypes.POINTER(_MpCorrectionC)),
      ("done", ctypes.c_bool),
  ]


@dataclasses.dataclass
class Correction:
  """A single correction.

  Attributes:
    type: The type of correction.
    text: The text associated with the correction.
  """

  type: CorrectionType
  text: str


@dataclasses.dataclass
class TextProofreaderResult:
  """Result of the text proofreader task.

  Attributes:
    proofread_text: The proofread text with corrections applied.
    corrections: List of unchanged, deleted, and inserted text segments.
    done: Whether the proofreading is done.
  """

  proofread_text: Optional[str]
  corrections: List[Correction]
  done: bool

  @classmethod
  def create_from_ctypes(
      cls,
      c_result: Union[
          _MpTextProofreaderResultC, _MpTextProofreaderStreamResultC
      ],
  ) -> "TextProofreaderResult":
    """Creates a `TextProofreaderResult` object from a C struct."""
    corrections = []
    if c_result.corrections and c_result.corrections_count > 0:
      for i in range(c_result.corrections_count):
        c_correction = c_result.corrections[i]
        corrections.append(
            Correction(
                type=CorrectionType(c_correction.type),
                text=(
                    c_correction.text.decode("utf-8")
                    if c_correction.text
                    else ""
                ),
            )
        )

    if isinstance(c_result, _MpTextProofreaderResultC):
      proofread_text = (
          c_result.proofread_text.decode("utf-8")
          if c_result.proofread_text is not None
          else None
      )
      done = True
    else:
      proofread_text = (
          c_result.chunk.decode("utf-8")
          if c_result.chunk is not None
          else None
      )
      done = c_result.done

    return TextProofreaderResult(
        proofread_text=proofread_text,
        corrections=corrections,
        done=done,
    )


_ProofreaderCallback = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.POINTER(_MpTextProofreaderStreamResultC),
    ctypes.c_char_p,
)


_CTYPES_SIGNATURES = (
    mediapipe_c_utils.CStatusFunction(
        "MpTextProofreaderCreate",
        [
            ctypes.POINTER(_MpTextProofreaderOptionsC),
            ctypes.POINTER(ctypes.c_void_p),
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        "MpTextProofreaderProofread",
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.POINTER(_MpTextProofreaderResultC)),
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        "MpTextProofreaderProofreadStreaming",
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            _ProofreaderCallback,
            ctypes.c_void_p,
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        "MpTextProofreaderClose",
        [
            ctypes.c_void_p,
        ],
    ),
    mediapipe_c_utils.CFunction(
        "MpTextProofreaderCloseResult",
        [ctypes.POINTER(_MpTextProofreaderResultC)],
        None,
    ),
)


class TextProofreader:
  """Performs proofreading on text."""

  _lib: serial_dispatcher.SerialDispatcher
  _proofreader_handle: ctypes.c_void_p

  def __init__(
      self, lib: serial_dispatcher.SerialDispatcher, handle: ctypes.c_void_p
  ):
    self._lib = lib
    self._proofreader_handle = handle
    self._async_result_callback: Optional[
        Callable[[Optional[TextProofreaderResult], Optional[str]], None]
    ] = None

    def callback_wrapper(
        unused_user_data: int,
        result_ptr: ctypes.POINTER(_MpTextProofreaderStreamResultC),
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
        c_result = cast(_MpTextProofreaderStreamResultC, result_ptr.contents)
        python_result = TextProofreaderResult.create_from_ctypes(c_result)
        self._async_result_callback(python_result, None)

        if c_result.done:
          is_done = True
      else:
        # Should not happen
        self._async_result_callback(None, "Unknown error")
        is_done = True

      if is_done:
        self._async_result_callback = None

    self._c_callback = _ProofreaderCallback(callback_wrapper)

  @classmethod
  def create_from_model_path(cls, model_path: str) -> "TextProofreader":
    """Creates a `TextProofreader` object from model path.

    Args:
      model_path: Path to the model.

    Returns:
      `TextProofreader` object that's created from the model file and the
      default `TextProofreaderOptions`.

    Raises:
      ValueError: If failed to create `TextProofreader` object from the provided
        file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = base_options_module.BaseOptions(model_asset_path=model_path)
    options = TextProofreaderOptions(base_options=base_options)
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(
      cls, options: TextProofreaderOptions
  ) -> "TextProofreader":
    """Creates the `TextProofreader` object from text proofreader options.

    Args:
      options: Options for the text proofreader task.

    Returns:
      `TextProofreader` object that's created from `options`.

    Raises:
      ValueError: If failed to create `TextProofreader` object from
        `TextProofreaderOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """
    lib = mediapipe_c_bindings.load_shared_library(_CTYPES_SIGNATURES)

    ctypes_options = options.to_ctypes()

    proofreader_handle = ctypes.c_void_p()
    lib.MpTextProofreaderCreate(
        ctypes.byref(ctypes_options),
        ctypes.byref(proofreader_handle),
    )
    return TextProofreader(lib=lib, handle=proofreader_handle)

  def proofread(self, text: str) -> TextProofreaderResult:
    """Performs proofreading on the input `text`.

    Args:
      text: The input text.

    Returns:
      A `TextProofreaderResult` object that contains the proofread text.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If proofreading failed to run.
    """
    result_ptr = ctypes.POINTER(_MpTextProofreaderResultC)()

    self._lib.MpTextProofreaderProofread(
        self._proofreader_handle,
        text.encode("utf-8"),
        ctypes.byref(result_ptr),
    )

    ctypes_result = cast(_MpTextProofreaderResultC, result_ptr.contents)
    python_result = TextProofreaderResult.create_from_ctypes(ctypes_result)
    self._lib.MpTextProofreaderCloseResult(result_ptr)
    return python_result

  def proofread_async(
      self,
      text: str,
      result_callback: Callable[
          [Optional[TextProofreaderResult], Optional[str]], None
      ],
  ) -> None:
    """Performs proofreading on the input `text` asynchronously.

    Args:
      text: The input text.
      result_callback: A callback function that will be called with the
        TextProofreaderResult and an error message (if any).

    Raises:
      ValueError: If any of the input arguments is invalid, or if
        proofread_async is called before the previous call has finished.
      RuntimeError: If proofreading failed to run.
    """
    if self._async_result_callback:
      raise ValueError(
          "proofread_async is already running. Concurrent calls are not"
          " supported."
      )
    self._async_result_callback = result_callback
    self._lib.MpTextProofreaderProofreadStreaming(
        self._proofreader_handle,
        text.encode("utf-8"),
        self._c_callback,
        None,
    )

  def close(self):
    """Shuts down the MediaPipe task instance."""
    if self._proofreader_handle:
      self._lib.MpTextProofreaderClose(self._proofreader_handle)
      self._proofreader_handle = None
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
