# pytype: skip-file
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
"""MediaPipe universal embedder task."""

import ctypes
import dataclasses
import enum
from typing import List, Optional

from mediapipe.tasks.python.components.containers import embedding_result as embedding_result_module
from mediapipe.tasks.python.components.containers import embedding_result_c as embedding_result_c_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import base_options_c as base_options_c_module
from mediapipe.tasks.python.core import mediapipe_c_bindings
from mediapipe.tasks.python.core import mediapipe_c_utils
from mediapipe.tasks.python.core import serial_dispatcher

UniversalEmbedderResult = embedding_result_module.EmbeddingResult


class ActivationDataType(enum.IntEnum):
  """Activation data type for model execution."""

  DEFAULT = 0
  FLOAT32 = 1
  FLOAT16 = 2
  INT16 = 3
  INT8 = 4


class _MpUniversalEmbedderOptionsC(ctypes.Structure):
  """C struct for MpUniversalEmbedderOptions."""

  _fields_ = [
      ('base_options', base_options_c_module.MpBaseOptionsC),
      ('l2_normalize', ctypes.c_bool),
      ('max_input_length', ctypes.c_int),
      ('vision_tokens_per_image', ctypes.c_int),
      ('activation_data_type', ctypes.c_int),
      ('cache_dir', ctypes.c_char_p),
  ]


_CTYPES_SIGNATURES = (
    mediapipe_c_utils.CStatusFunction(
        'MpUniversalEmbedderCreate',
        [
            ctypes.POINTER(_MpUniversalEmbedderOptionsC),
            ctypes.POINTER(ctypes.c_void_p),
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpUniversalEmbedderEmbedText',
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(embedding_result_c_module.MpEmbeddingResultC),
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpUniversalEmbedderEmbedImage',
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(embedding_result_c_module.MpEmbeddingResultC),
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpUniversalEmbedderEmbedAudio',
        [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(embedding_result_c_module.MpEmbeddingResultC),
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpUniversalEmbedderClose',
        [
            ctypes.c_void_p,
        ],
    ),
    mediapipe_c_utils.CFunction(
        'MpUniversalEmbedderCloseResult',
        [ctypes.POINTER(embedding_result_c_module.MpEmbeddingResultC)],
        None,
    ),
)


@dataclasses.dataclass
class UniversalEmbedderOptions:
  """Options for the universal embedder task."""

  base_options: base_options_module.BaseOptions
  l2_normalize: Optional[bool] = None
  max_input_length: Optional[int] = None
  vision_tokens_per_image: Optional[int] = None
  activation_data_type: Optional[ActivationDataType] = None
  cache_dir: Optional[str] = None

  def to_ctypes(self) -> _MpUniversalEmbedderOptionsC:
    """Generates a ctypes MpUniversalEmbedderOptionsC object."""
    return _MpUniversalEmbedderOptionsC(
        base_options=self.base_options.to_ctypes(),
        l2_normalize=self.l2_normalize
        if self.l2_normalize is not None
        else False,
        max_input_length=self.max_input_length
        if self.max_input_length is not None
        else 0,
        vision_tokens_per_image=self.vision_tokens_per_image
        if self.vision_tokens_per_image is not None
        else 0,
        activation_data_type=self.activation_data_type.value
        if self.activation_data_type is not None
        else ActivationDataType.DEFAULT.value,
        cache_dir=self.cache_dir.encode('utf-8') if self.cache_dir else None,
    )


class UniversalEmbedder:
  """Class that performs embedding extraction on multi-modal inputs."""

  _lib: serial_dispatcher.SerialDispatcher
  _handle: ctypes.c_void_p

  def __init__(
      self,
      lib: serial_dispatcher.SerialDispatcher,
      handle: ctypes.c_void_p,
  ) -> None:
    self._lib = lib
    self._embedder_handle = handle

  @classmethod
  def create_from_options(
      cls, options: UniversalEmbedderOptions
  ) -> 'UniversalEmbedder':
    """Creates the `UniversalEmbedder` object from options."""
    lib = mediapipe_c_bindings.load_shared_library(_CTYPES_SIGNATURES)
    ctypes_options = options.to_ctypes()
    embedder_handle = ctypes.c_void_p()
    lib.MpUniversalEmbedderCreate(
        ctypes.byref(ctypes_options), ctypes.byref(embedder_handle)
    )
    return UniversalEmbedder(lib=lib, handle=embedder_handle)

  def embed_text(self, text: str) -> UniversalEmbedderResult:
    """Extracts embedding from input text."""
    result_c = embedding_result_c_module.MpEmbeddingResultC()
    self._lib.MpUniversalEmbedderEmbedText(
        self._embedder_handle,
        text.encode('utf-8'),
        ctypes.byref(result_c),
    )
    result = embedding_result_module.EmbeddingResult.from_ctypes(result_c)
    self._lib.MpUniversalEmbedderCloseResult(ctypes.byref(result_c))
    return result

  def embed_image(self, image_bytes: bytes) -> UniversalEmbedderResult:
    """Extracts embedding from raw image bytes."""
    result_c = embedding_result_c_module.MpEmbeddingResultC()
    self._lib.MpUniversalEmbedderEmbedImage(
        self._embedder_handle,
        ctypes.c_char_p(image_bytes),
        len(image_bytes),
        ctypes.byref(result_c),
    )
    result = embedding_result_module.EmbeddingResult.from_ctypes(result_c)
    self._lib.MpUniversalEmbedderCloseResult(ctypes.byref(result_c))
    return result

  def embed_audio(self, audio_data: List[float]) -> UniversalEmbedderResult:
    """Extracts embedding from float audio samples."""
    result_c = embedding_result_c_module.MpEmbeddingResultC()
    audio_arr = (ctypes.c_float * len(audio_data))(*audio_data)
    self._lib.MpUniversalEmbedderEmbedAudio(
        self._embedder_handle,
        audio_arr,
        len(audio_data),
        ctypes.byref(result_c),
    )
    result = embedding_result_module.EmbeddingResult.from_ctypes(result_c)
    self._lib.MpUniversalEmbedderCloseResult(ctypes.byref(result_c))
    return result

  def close(self) -> None:
    """Shuts down and releases the UniversalEmbedder."""
    if not self._embedder_handle:
      return
    self._lib.MpUniversalEmbedderClose(self._embedder_handle)
    self._embedder_handle = None
    self._lib.close()

  def __enter__(self) -> 'UniversalEmbedder':
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    self.close()

  def __del__(self) -> None:
    self.close()
