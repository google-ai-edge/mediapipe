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
import struct
from typing import List, Optional, Sequence, Union
import wave

from mediapipe.tasks.python.components.containers import embedding_result as embedding_result_module
from mediapipe.tasks.python.components.containers import embedding_result_c as embedding_result_c_module
from mediapipe.tasks.python.components.utils import cosine_similarity
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


@dataclasses.dataclass
class TextPart:
  """A text part in a multimodal content block."""

  text: str


@dataclasses.dataclass
class ImagePart:
  """An image part in a multimodal content block."""

  image_bytes: Optional[bytes] = None
  file_path: Optional[str] = None


@dataclasses.dataclass
class AudioPart:
  """An audio part in a multimodal content block."""

  audio_data: Optional[List[float]] = None
  audio_path: Optional[str] = None


ContentPart = Union[TextPart, ImagePart, AudioPart]


def _read_wav_mono_float(audio_path: str) -> List[float]:
  """Reads a mono 16-bit PCM WAV file into normalized float samples."""
  with wave.open(audio_path, 'rb') as wf:
    channels = wf.getnchannels()
    if channels != 1:
      raise ValueError(
          f'Only mono (1 channel) WAV audio is supported, got {channels}.'
      )
    sample_width = wf.getsampwidth()
    if sample_width != 2:
      raise ValueError(
          f'Only 16-bit PCM WAV audio is supported, got {sample_width * 8}-bit.'
      )
    comptype = wf.getcomptype()
    if comptype != 'NONE':
      raise ValueError(
          f'Only uncompressed PCM WAV audio is supported, got {comptype!r}.'
      )
    raw_frames = wf.readframes(wf.getnframes())
  sample_count = len(raw_frames) // 2
  samples_int16 = struct.unpack_from(f'<{sample_count}h', raw_frames)
  return [s / 32768.0 for s in samples_int16]


class _PartType(enum.IntEnum):
  TEXT = 1
  IMAGE = 2
  AUDIO = 3


class _MpUniversalEmbedderImageSliceC(ctypes.Structure):
  _fields_ = [
      ('bytes', ctypes.c_void_p),
      ('size', ctypes.c_int),
  ]


class _MpUniversalEmbedderAudioSliceC(ctypes.Structure):
  _fields_ = [
      ('samples', ctypes.POINTER(ctypes.c_float)),
      ('size', ctypes.c_int),
  ]


class _MpUniversalEmbedderPartValueC(ctypes.Union):
  _fields_ = [
      ('text', ctypes.c_char_p),
      ('image', _MpUniversalEmbedderImageSliceC),
      ('audio', _MpUniversalEmbedderAudioSliceC),
  ]


class _MpUniversalEmbedderPartC(ctypes.Structure):
  _fields_ = [
      ('type', ctypes.c_int),
      ('value', _MpUniversalEmbedderPartValueC),
  ]


class _MpUniversalEmbedderOptionsC(ctypes.Structure):
  """C struct for MpUniversalEmbedderOptions."""

  _fields_ = [
      ('base_options', base_options_c_module.MpBaseOptionsC),
      ('l2_normalize', ctypes.c_bool),
      ('max_input_length', ctypes.c_int),
      ('vision_tokens_per_image', ctypes.c_int),
      ('activation_data_type', ctypes.c_int),
      ('cache_dir', ctypes.c_char_p),
      ('text_delegate', ctypes.c_int),
      ('vision_delegate', ctypes.c_int),
      ('audio_delegate', ctypes.c_int),
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
        'MpUniversalEmbedderEmbedContent',
        [
            ctypes.c_void_p,
            ctypes.POINTER(_MpUniversalEmbedderPartC),
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
  l2_normalize: bool = True
  max_input_length: Optional[int] = None
  vision_tokens_per_image: Optional[int] = None
  activation_data_type: Optional[ActivationDataType] = None
  cache_dir: Optional[str] = None
  text_delegate: Optional[base_options_module.BaseOptions.Delegate] = None
  vision_delegate: Optional[base_options_module.BaseOptions.Delegate] = None
  audio_delegate: Optional[base_options_module.BaseOptions.Delegate] = None

  def to_ctypes(self) -> _MpUniversalEmbedderOptionsC:
    """Generates a ctypes MpUniversalEmbedderOptionsC object."""
    return _MpUniversalEmbedderOptionsC(
        base_options=self.base_options.to_ctypes(),
        l2_normalize=bool(self.l2_normalize)
        if self.l2_normalize is not None
        else True,
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
        text_delegate=self.text_delegate.value
        if self.text_delegate is not None
        else 0,
        vision_delegate=self.vision_delegate.value
        if self.vision_delegate is not None
        else 0,
        audio_delegate=self.audio_delegate.value
        if self.audio_delegate is not None
        else 0,
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

  def embed_content(
      self, parts: Sequence[ContentPart]
  ) -> UniversalEmbedderResult:
    """Extracts a joint embedding from multimodal content parts.

    Args:
      parts: Sequence of multimodal content parts to embed together.

    Returns:
      An `EmbeddingResult` representing the joint embedding.
    """
    parts_arr = (_MpUniversalEmbedderPartC * len(parts))()
    refs = []
    for i, part in enumerate(parts):
      if isinstance(part, TextPart):
        encoded = part.text.encode('utf-8')
        refs.append(encoded)
        parts_arr[i].type = _PartType.TEXT.value
        parts_arr[i].value.text = encoded
      elif isinstance(part, ImagePart):
        img_bytes = part.image_bytes
        if img_bytes is None and part.file_path:
          with open(part.file_path, 'rb') as f:
            img_bytes = f.read()
        if img_bytes is None:
          raise ValueError('ImagePart requires `image_bytes` or `file_path`.')
        buf = ctypes.create_string_buffer(img_bytes, len(img_bytes))
        refs.append(buf)
        parts_arr[i].type = _PartType.IMAGE.value
        parts_arr[i].value.image.bytes = ctypes.cast(buf, ctypes.c_void_p)
        parts_arr[i].value.image.size = len(img_bytes)
      elif isinstance(part, AudioPart):
        samples = part.audio_data
        if samples is None and part.audio_path:
          samples = _read_wav_mono_float(part.audio_path)
        if samples is None:
          raise ValueError('AudioPart requires `audio_data` or `audio_path`.')
        audio_arr = (ctypes.c_float * len(samples))(*samples)
        refs.append(audio_arr)
        parts_arr[i].type = _PartType.AUDIO.value
        parts_arr[i].value.audio.samples = audio_arr
        parts_arr[i].value.audio.size = len(samples)
      else:
        raise TypeError(f'Unsupported content part: {type(part)!r}')

    result_c = embedding_result_c_module.MpEmbeddingResultC()
    self._lib.MpUniversalEmbedderEmbedContent(
        self._embedder_handle,
        parts_arr,
        len(parts),
        ctypes.byref(result_c),
    )
    result = embedding_result_module.EmbeddingResult.from_ctypes(result_c)
    self._lib.MpUniversalEmbedderCloseResult(ctypes.byref(result_c))
    return result

  @classmethod
  def cosine_similarity(
      cls,
      u: embedding_result_module.Embedding,
      v: embedding_result_module.Embedding,
  ) -> float:
    """Utility function to compute cosine similarity between two embedding entries.

    May return an InvalidArgumentError if e.g. the feature vectors are
    of different types (quantized vs. float), have different sizes, or have a
    an L2-norm of 0.

    Args:
      u: An embedding entry.
      v: An embedding entry.

    Returns:
      The cosine similarity for the two embeddings.

    Raises:
      ValueError: May return an error if e.g. the feature vectors are of
        different types (quantized vs. float), have different sizes, or have
        an L2-norm of 0.
    """
    return cosine_similarity.cosine_similarity(u, v)

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
