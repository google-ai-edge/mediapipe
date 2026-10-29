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
"""MediaPipe AudioTranscriber Python task."""

from typing import Sequence
import ctypes
import dataclasses
import enum
import os

import numpy as np

from mediapipe.tasks.python.core import base_options as base_options_lib
from mediapipe.tasks.python.core import base_options_c
from mediapipe.tasks.python.core import mediapipe_c_bindings
from mediapipe.tasks.python.core import mediapipe_c_utils

_BaseOptions = base_options_lib.BaseOptions


class ModelArchitecture(enum.IntEnum):
  AUTO = 0
  OMNI = 1


class _MpTranscriptionSegmentC(ctypes.Structure):
  _fields_ = [
      ('transcript', ctypes.c_char_p),
      ('is_final', ctypes.c_bool),
      ('from_timestamp_ms', ctypes.c_int64),
      ('to_timestamp_ms', ctypes.c_int64),
  ]


class _MpAudioTranscriberResultC(ctypes.Structure):
  _fields_ = [
      ('segments', ctypes.POINTER(_MpTranscriptionSegmentC)),
      ('segments_count', ctypes.c_uint32),
  ]


class _MpAudioTranscriberOptionsC(ctypes.Structure):
  _fields_ = [
      ('base_options', base_options_c.MpBaseOptionsC),
      ('sample_rate_hz', ctypes.c_int),
      ('enable_interim_results', ctypes.c_bool),
      ('model_architecture', ctypes.c_int),
      ('prompt', ctypes.c_char_p),
      ('cache_dir', ctypes.c_char_p),
  ]


_CTYPES_SIGNATURES = (
    mediapipe_c_utils.CFunction(
        'MpAudioTranscriberCreate',
        [
            ctypes.POINTER(_MpAudioTranscriberOptionsC),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_int,
    ),
    mediapipe_c_utils.CFunction(
        'MpAudioTranscriberTranscribe',
        [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_bool,
            ctypes.POINTER(_MpAudioTranscriberResultC),
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_int,
    ),
    mediapipe_c_utils.CFunction(
        'MpAudioTranscriberFinishStream',
        [
            ctypes.c_void_p,
            ctypes.POINTER(_MpAudioTranscriberResultC),
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_int,
    ),
    mediapipe_c_utils.CFunction(
        'MpAudioTranscriberPrewarm',
        [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_int,
    ),
    mediapipe_c_utils.CFunction(
        'MpAudioTranscriberCloseResult',
        [
            ctypes.POINTER(_MpAudioTranscriberResultC),
        ],
        None,
    ),
    mediapipe_c_utils.CFunction(
        'MpAudioTranscriberClose',
        [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_int,
    ),
)


def _get_lib() -> ctypes.CDLL:
  """Loads and configures the MediaPipe shared library."""
  os.environ.setdefault('PRODKERNEL_API_DISABLE_RESTARTABLE_SEQUENCES', '1')
  return mediapipe_c_bindings.load_raw_library(_CTYPES_SIGNATURES)


def _check_status(lib: ctypes.CDLL, status: int, err_ptr: ctypes.c_char_p):
  if status != 0:
    msg = (
        err_ptr.value.decode('utf-8', errors='replace')
        if err_ptr.value
        else f'MpStatus={status}'
    )
    if err_ptr.value:
      lib.MpErrorFree(err_ptr)
    raise RuntimeError(f'AudioTranscriber C API error ({status}): {msg}')


@dataclasses.dataclass(frozen=True)
class TranscriptionSegment:
  """A single transcribed segment with start and end timestamps.

  Attributes:
    transcript: The recognized text for this audio segment.
    is_final: Whether this segment is finalized or an interim hypothesis.
    from_timestamp_ms: Segment start time in milliseconds relative to stream
      start.
    to_timestamp_ms: Segment end time in milliseconds relative to stream start.
  """

  transcript: str
  is_final: bool
  from_timestamp_ms: int
  to_timestamp_ms: int


@dataclasses.dataclass(frozen=True)
class AudioTranscriberResult:
  """Transcription result returned by `AudioTranscriber`.

  Attributes:
    segments: Ordered list of transcribed segments produced for the input audio.
  """

  segments: list[TranscriptionSegment]

  @property
  def text(self) -> str:
    """Returns the concatenated transcript across all non-empty segments."""
    return ' '.join(
        s.transcript.strip() for s in self.segments if s.transcript.strip()
    )


@dataclasses.dataclass
class AudioTranscriberOptions:
  """Options for MediaPipe AudioTranscriber task.

  Attributes:
    base_options: Base options for the MediaPipe task (model path and delegate).
    sample_rate_hz: Input PCM audio sample rate in Hz.
    enable_interim_results: Whether to emit interim hypotheses during streaming.
    model_architecture: ASR model family routing (`AUTO`, `OMNI`, or
      `TINY_GEMMA`).
    prompt: Optional decoding prompt override; `None` uses the model default.
    cache_dir: Optional directory path for compiled weight/program caches;
      `None` disables disk caching.
  """

  base_options: _BaseOptions
  sample_rate_hz: int = 16000
  enable_interim_results: bool = False
  model_architecture: ModelArchitecture = ModelArchitecture.AUTO
  prompt: str | None = None
  cache_dir: str | None = None


class AudioTranscriber:
  """Performs streaming and unary speech-to-text transcription."""

  def __init__(self, handle: ctypes.c_void_p):
    """Initializes an `AudioTranscriber` wrapping a C API handle.

    Args:
      handle: Opaque C pointer created by `MpAudioTranscriberCreate`.
    """
    self._lib = _get_lib()
    self._handle = handle

  @classmethod
  def create_from_options(
      cls, options: AudioTranscriberOptions
  ) -> 'AudioTranscriber':
    """Creates an AudioTranscriber instance from `AudioTranscriberOptions`."""
    lib = _get_lib()
    c_base = options.base_options.to_ctypes()
    prompt_bytes = options.prompt.encode('utf-8') if options.prompt else None
    cache_bytes = (
        options.cache_dir.encode('utf-8') if options.cache_dir else None
    )
    c_opts = _MpAudioTranscriberOptionsC(
        base_options=c_base,
        sample_rate_hz=options.sample_rate_hz,
        enable_interim_results=options.enable_interim_results,
        model_architecture=int(options.model_architecture),
        prompt=prompt_bytes,
        cache_dir=cache_bytes,
    )
    handle = ctypes.c_void_p()
    err = ctypes.c_char_p()
    status = lib.MpAudioTranscriberCreate(
        ctypes.byref(c_opts), ctypes.byref(handle), ctypes.byref(err)
    )
    _check_status(lib, status, err)
    return cls(handle)

  @classmethod
  def create_from_model_path(
      cls,
      model_path: str,
      delegate: _BaseOptions.Delegate = _BaseOptions.Delegate.CPU,
      sample_rate_hz: int = 16000,
  ) -> 'AudioTranscriber':
    """Creates an AudioTranscriber from a `.litertlm` model path."""
    opts = AudioTranscriberOptions(
        base_options=_BaseOptions(
            model_asset_path=model_path, delegate=delegate
        ),
        sample_rate_hz=sample_rate_hz,
    )
    return cls.create_from_options(opts)

  def prewarm(self) -> None:
    """Runs a dummy warm-up pass to initialize execution kernels."""
    err = ctypes.c_char_p()
    status = self._lib.MpAudioTranscriberPrewarm(
        self._handle, ctypes.byref(err)
    )
    _check_status(self._lib, status, err)

  def transcribe(
      self,
      audio_samples: Sequence[float] | np.ndarray,
      end_of_stream: bool = True,
  ) -> AudioTranscriberResult:
    """Transcribes mono PCM float32 audio samples in `[-1.0, 1.0]`.

    Args:
      audio_samples: Sequence or 1D NumPy array of mono float32 PCM samples.
      end_of_stream: Whether this audio chunk concludes the current stream.

    Returns:
      An `AudioTranscriberResult` containing recognized segments.
    """
    arr = np.ascontiguousarray(audio_samples, dtype=np.float32)
    c_samples = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_res = _MpAudioTranscriberResultC()
    err = ctypes.c_char_p()
    status = self._lib.MpAudioTranscriberTranscribe(
        self._handle,
        c_samples,
        arr.size,
        end_of_stream,
        ctypes.byref(c_res),
        ctypes.byref(err),
    )
    _check_status(self._lib, status, err)
    return self._extract_result(c_res)

  def finish_stream(self) -> AudioTranscriberResult:
    """Flushes any buffered audio and finalizes the current stream."""
    c_res = _MpAudioTranscriberResultC()
    err = ctypes.c_char_p()
    status = self._lib.MpAudioTranscriberFinishStream(
        self._handle, ctypes.byref(c_res), ctypes.byref(err)
    )
    _check_status(self._lib, status, err)
    return self._extract_result(c_res)

  def _extract_result(
      self, c_res: _MpAudioTranscriberResultC
  ) -> AudioTranscriberResult:
    """Converts and frees a C transcription result struct."""
    segments: list[TranscriptionSegment] = []
    try:
      for i in range(c_res.segments_count):
        seg = c_res.segments[i]
        text = (
            seg.transcript.decode('utf-8', errors='replace')
            if seg.transcript
            else ''
        )
        segments.append(
            TranscriptionSegment(
                transcript=text,
                is_final=bool(seg.is_final),
                from_timestamp_ms=int(seg.from_timestamp_ms),
                to_timestamp_ms=int(seg.to_timestamp_ms),
            )
        )
    finally:
      self._lib.MpAudioTranscriberCloseResult(ctypes.byref(c_res))
    return AudioTranscriberResult(segments=segments)

  def close(self) -> None:
    """Closes the transcriber and releases native C++ resources."""
    if self._handle:
      err = ctypes.c_char_p()
      status = self._lib.MpAudioTranscriberClose(
          self._handle, ctypes.byref(err)
      )
      self._handle = ctypes.c_void_p()
      _check_status(self._lib, status, err)

  def __enter__(self) -> 'AudioTranscriber':
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    self.close()
