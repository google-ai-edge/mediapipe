# Copyright 2026 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MediaPipe audio embedder task."""

import ctypes
import dataclasses
from typing import List, Optional

from mediapipe.tasks.python.audio.core import audio_record
from mediapipe.tasks.python.components.containers import audio_data
from mediapipe.tasks.python.components.containers import audio_data_c
from mediapipe.tasks.python.components.containers import embedding_result as embedding_result_module
from mediapipe.tasks.python.components.containers import embedding_result_c as embedding_result_c_module
from mediapipe.tasks.python.components.utils import cosine_similarity
from mediapipe.tasks.python.core import base_options as base_options_lib
from mediapipe.tasks.python.core import base_options_c
from mediapipe.tasks.python.core import mediapipe_c_bindings
from mediapipe.tasks.python.core import mediapipe_c_utils
from mediapipe.tasks.python.core import serial_dispatcher
from mediapipe.tasks.python.core.optional_dependencies import doc_controls

AudioEmbedderResult = embedding_result_module.EmbeddingResult
_AudioData = audio_data.AudioData
_BaseOptions = base_options_lib.BaseOptions


class MpAudioEmbedderResultC(ctypes.Structure):
  """The C representation of a list of audio embedding results."""

  _fields_ = [
      (
          'results',
          ctypes.POINTER(embedding_result_c_module.MpEmbeddingResultC),
      ),
      ('results_count', ctypes.c_int),
  ]


class _MpEmbedderOptionsC(ctypes.Structure):
  """C struct for MpEmbedderOptions."""

  _fields_ = [
      ('l2_normalize', ctypes.c_bool),
      ('quantize', ctypes.c_bool),
  ]


class MpAudioEmbedderOptionsC(ctypes.Structure):
  """The audio embedder options used in the C API."""

  _fields_ = [
      ('base_options', base_options_c.MpBaseOptionsC),
      ('embedder_options', _MpEmbedderOptionsC),
  ]

  @classmethod
  @doc_controls.do_not_generate_docs
  def from_c_options(
      cls,
      base_options: base_options_c.MpBaseOptionsC,
      embedder_options: _MpEmbedderOptionsC,
  ) -> 'MpAudioEmbedderOptionsC':
    """Creates an MpAudioEmbedderOptionsC object from the given options."""
    return cls(
        base_options=base_options,
        embedder_options=embedder_options,
    )


_CTYPES_SIGNATURES = (
    mediapipe_c_utils.CStatusFunction(
        'MpAudioEmbedderCreate',
        (
            ctypes.POINTER(MpAudioEmbedderOptionsC),
            ctypes.POINTER(ctypes.c_void_p),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpAudioEmbedderEmbed',
        (
            ctypes.c_void_p,
            ctypes.POINTER(audio_data_c.MpAudioDataC),
            ctypes.POINTER(MpAudioEmbedderResultC),
        ),
    ),
    mediapipe_c_utils.CFunction(
        'MpAudioEmbedderCloseResult',
        [ctypes.POINTER(MpAudioEmbedderResultC)],
        None,
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpAudioEmbedderClose',
        (ctypes.c_void_p,),
    ),
)


@dataclasses.dataclass
class AudioEmbedderOptions:
  """Options for the audio embedder task.

  Attributes:
    base_options: Base options for the audio embedder task.
    l2_normalize: Whether to normalize the returned feature vector with L2 norm.
    quantize: Whether the returned embedding should be quantized to bytes via
      scalar quantization. Embeddings are implicitly assumed to be unit-norm and
      therefore any dimension is guaranteed to have a value in [-1.0, 1.0]. Use
      the l2_normalize option if this is not the case.
  """

  base_options: _BaseOptions
  l2_normalize: Optional[bool] = None
  quantize: Optional[bool] = None


class AudioEmbedder:
  """Class that performs embedding extraction on audio clips.

  This API expects a .litertlm model and uses LiteRT-LM for inference.
  """

  _lib: serial_dispatcher.SerialDispatcher
  _handle: ctypes.c_void_p

  def __init__(
      self,
      lib: serial_dispatcher.SerialDispatcher,
      handle: ctypes.c_void_p,
  ):
    """Initializes the AudioEmbedder instance.

    Args:
      lib: The serial dispatcher for the audio embedder task.
      handle: The handle to the audio embedder task.
    """
    self._lib = lib
    self._handle = handle

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'AudioEmbedder':
    """Creates an `AudioEmbedder` object from a .litertlm model and the default `AudioEmbedderOptions`.

    Args:
      model_path: Path to the model.

    Returns:
      `AudioEmbedder` object that's created from the model file and the
      default `AudioEmbedderOptions`.

    Raises:
      ValueError: If failed to create `AudioEmbedder` object from the provided
        file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = AudioEmbedderOptions(base_options=base_options)
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(
      cls, options: AudioEmbedderOptions
  ) -> 'AudioEmbedder':
    """Creates the `AudioEmbedder` object from audio embedder options.

    Args:
      options: Options for the audio embedder task.

    Returns:
      `AudioEmbedder` object that's created from `options`.

    Raises:
      ValueError: If failed to create `AudioEmbedder` object from
        `AudioEmbedderOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """
    lib = mediapipe_c_bindings.load_shared_library(_CTYPES_SIGNATURES)

    ctypes_options = MpAudioEmbedderOptionsC.from_c_options(
        base_options=options.base_options.to_ctypes(),
        embedder_options=_MpEmbedderOptionsC(
            l2_normalize=bool(options.l2_normalize),
            quantize=bool(options.quantize),
        ),
    )

    embedder_handle_ptr = ctypes.c_void_p()
    lib.MpAudioEmbedderCreate(  # pyrefly: ignore[missing-attribute]
        ctypes.byref(ctypes_options), ctypes.byref(embedder_handle_ptr)
    )
    return AudioEmbedder(
        lib=lib,
        handle=embedder_handle_ptr,
    )

  def embed(self, audio_clip: _AudioData) -> List[AudioEmbedderResult]:
    """Performs embedding extraction on the provided audio clips.

    The audio clip is represented as a MediaPipe AudioData. The method accepts
    audio clips with various length and audio sample rate. It's required to
    provide the corresponding audio sample rate within the `AudioData` object.

    Args:
      audio_clip: MediaPipe AudioData.

    Returns:
      A list of `AudioEmbedderResult` objects that contain the embedding
      results.

    Raises:
      ValueError: If any of the input arguments is invalid, such as the sample
        rate is not provided in the `AudioData` object.
      RuntimeError: If audio embedding extraction failed to run.
    """
    if not audio_clip.audio_format.sample_rate:
      raise ValueError('Must provide the audio sample rate in audio data.')

    c_result = MpAudioEmbedderResultC()
    self._lib.MpAudioEmbedderEmbed(  # pyrefly: ignore[missing-attribute]
        self._handle,
        audio_clip.to_ctypes(),
        ctypes.byref(c_result),
    )
    py_result = [
        embedding_result_module.EmbeddingResult.from_ctypes(c_result.results[i])
        for i in range(c_result.results_count)
    ]
    self._lib.MpAudioEmbedderCloseResult(ctypes.byref(c_result))  # pyrefly: ignore[missing-attribute]
    return py_result

  def create_audio_record(
      self, num_channels: int, sample_rate: int, required_input_buffer_size: int
  ) -> audio_record.AudioRecord:
    """Creates an AudioRecord instance to record audio stream.

    The returned AudioRecord instance is initialized and client needs to call
    the appropriate method to start recording.

    Note that MediaPipe Audio tasks will up/down sample automatically to fit the
    sample rate required by the model.

    Args:
      num_channels: The number of audio channels.
      sample_rate: The audio sample rate.
      required_input_buffer_size: The required input buffer size in number of
        float elements.

    Returns:
      An AudioRecord instance.

    Raises:
      ValueError: If there's a problem creating the AudioRecord instance.
    """
    return audio_record.AudioRecord(
        num_channels, sample_rate, required_input_buffer_size
    )

  def close(self):
    """Shuts down the MediaPipe task instance."""
    if self._handle:
      self._lib.MpAudioEmbedderClose(self._handle)  # pyrefly: ignore[missing-attribute]
      self._handle = None  # pyrefly: ignore[bad-assignment]
      self._lib.close()

  @classmethod
  def cosine_similarity(
      cls,
      u: embedding_result_module.Embedding,
      v: embedding_result_module.Embedding,
  ) -> float:
    """Utility function to compute cosine similarity between two embedding entries.

    May return an InvalidArgumentError if e.g. the feature vectors are
    of different types (quantized vs. float), have different sizes, or have an
    L2-norm of 0.

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

  def __enter__(self):
    """Returns `self` upon entering the runtime context."""
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Shuts down the MediaPipe task instance on exit of the context manager.

    Args:
      exc_type: The exception type that caused the exit.
      exc_value: The exception value that caused the exit.
      traceback: The exception traceback that caused the exit.
    """
    del exc_type, exc_value, traceback  # Unused.
    self.close()

  def __del__(self):
    self.close()
