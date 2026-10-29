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
"""MediaPipe TextToSpeech Python task."""

from typing import Iterator
import ctypes
import dataclasses
import os

import numpy as np

from mediapipe.tasks.python.core import base_options as base_options_lib
from mediapipe.tasks.python.core import base_options_c
from mediapipe.tasks.python.core import mediapipe_c_bindings
from mediapipe.tasks.python.core import mediapipe_c_utils

_BaseOptions = base_options_lib.BaseOptions
_MP_OUT_OF_RANGE = 11


class _MpTextToSpeechVoiceInfoC(ctypes.Structure):
  _fields_ = [
      ('voice_id', ctypes.c_char_p),
      ('display_name', ctypes.c_char_p),
      ('locale', ctypes.c_char_p),
      ('sample_rate_hz', ctypes.c_int32),
      ('is_downloaded', ctypes.c_bool),
  ]


class _MpTextToSpeechVoiceListC(ctypes.Structure):
  _fields_ = [
      ('voices', ctypes.POINTER(_MpTextToSpeechVoiceInfoC)),
      ('num_voices', ctypes.c_size_t),
  ]


class _MpTextToSpeechOptionsC(ctypes.Structure):
  _fields_ = [
      ('base_options', base_options_c.MpBaseOptionsC),
      ('default_voice_id', ctypes.c_char_p),
      ('default_locale', ctypes.c_char_p),
      ('speaking_rate', ctypes.c_float),
      ('target_bucket', ctypes.c_int32),
      ('cache_directory', ctypes.c_char_p),
  ]


class _MpTextToSpeechSynthesisOptionsC(ctypes.Structure):
  _fields_ = [
      ('voice_id', ctypes.c_char_p),
      ('locale', ctypes.c_char_p),
  ]


class _MpTextToSpeechWordTimestampC(ctypes.Structure):
  _fields_ = [
      ('word', ctypes.c_char_p),
      ('start_char', ctypes.c_int32),
      ('end_char', ctypes.c_int32),
      ('start_sec', ctypes.c_float),
      ('end_sec', ctypes.c_float),
  ]


class _MpTextToSpeechAudioResultC(ctypes.Structure):
  _fields_ = [
      ('samples', ctypes.POINTER(ctypes.c_float)),
      ('num_samples', ctypes.c_size_t),
      ('sample_rate_hz', ctypes.c_int32),
      ('word_timestamps', ctypes.POINTER(_MpTextToSpeechWordTimestampC)),
      ('num_word_timestamps', ctypes.c_size_t),
  ]


_CTYPES_SIGNATURES = (
    mediapipe_c_utils.CFunction(
        'MpTextToSpeechCreate',
        [
            ctypes.POINTER(_MpTextToSpeechOptionsC),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_int,
    ),
    mediapipe_c_utils.CFunction(
        'MpTextToSpeechListVoices',
        [
            ctypes.c_void_p,
            ctypes.POINTER(_MpTextToSpeechVoiceListC),
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_int,
    ),
    mediapipe_c_utils.CFunction(
        'MpTextToSpeechVoiceListFree',
        [
            ctypes.POINTER(_MpTextToSpeechVoiceListC),
        ],
        None,
    ),
    mediapipe_c_utils.CFunction(
        'MpTextToSpeechSynthesize',
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(_MpTextToSpeechSynthesisOptionsC),
            ctypes.POINTER(_MpTextToSpeechAudioResultC),
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_int,
    ),
    mediapipe_c_utils.CFunction(
        'MpTextToSpeechFreeAudioResult',
        [
            ctypes.POINTER(_MpTextToSpeechAudioResultC),
        ],
        None,
    ),
    mediapipe_c_utils.CFunction(
        'MpTextToSpeechComputeWordTimestamps',
        [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.POINTER(_MpTextToSpeechAudioResultC),
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_int,
    ),
    mediapipe_c_utils.CFunction(
        'MpTextToSpeechStartStream',
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(_MpTextToSpeechSynthesisOptionsC),
            ctypes.c_bool,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_int,
    ),
    mediapipe_c_utils.CFunction(
        'MpTextToSpeechStreamReadNext',
        [
            ctypes.c_void_p,
            ctypes.POINTER(_MpTextToSpeechAudioResultC),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_bool),
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_int,
    ),
    mediapipe_c_utils.CFunction(
        'MpTextToSpeechStreamPushText',
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_int,
    ),
    mediapipe_c_utils.CFunction(
        'MpTextToSpeechStreamFinishInput',
        [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_int,
    ),
    mediapipe_c_utils.CFunction(
        'MpTextToSpeechStreamCancel',
        [ctypes.c_void_p],
        None,
    ),
    mediapipe_c_utils.CFunction(
        'MpTextToSpeechStreamClose',
        [ctypes.c_void_p],
        None,
    ),
    mediapipe_c_utils.CFunction(
        'MpTextToSpeechCancel',
        [ctypes.c_void_p],
        None,
    ),
    mediapipe_c_utils.CFunction(
        'MpTextToSpeechClose',
        [ctypes.c_void_p],
        None,
    ),
    mediapipe_c_utils.CFunction(
        'MpTextToSpeechFreeError',
        [ctypes.c_char_p],
        None,
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
      lib.MpTextToSpeechFreeError(err_ptr)
    raise RuntimeError(f'TextToSpeech C API error ({status}): {msg}')


@dataclasses.dataclass(frozen=True)
class VoiceInfo:
  """Metadata for a speaker voice available in a TTS model bundle.

  Attributes:
    voice_id: Unique identifier of the voice (e.g. `af_heart`).
    display_name: Human-readable display name of the voice.
    locale: BCP-47 locale tag supported by the voice (e.g. `en-US`).
    sample_rate_hz: Native output PCM sample rate in Hz.
    is_downloaded: Whether the voice weights are locally resident in the bundle.
  """

  voice_id: str
  display_name: str
  locale: str
  sample_rate_hz: int
  is_downloaded: bool = True


@dataclasses.dataclass(frozen=True)
class WordTimestamp:
  """Word-level timing alignment mapping an input word to audio seconds.

  Attributes:
    word: Original surface word token from the input text.
    start_char: Zero-based character start offset in the input text.
    end_char: Zero-based character end offset (exclusive) in the input text.
    start_sec: Estimated start time in seconds within the synthesized audio.
    end_sec: Estimated end time in seconds within the synthesized audio.
  """

  word: str
  start_char: int
  end_char: int
  start_sec: float
  end_sec: float


def _extract_word_timestamps_from_c(
    c_res: _MpTextToSpeechAudioResultC,
) -> tuple[WordTimestamp, ...]:
  """Extracts `WordTimestamp` entries from a `_MpTextToSpeechAudioResultC`."""
  if c_res.num_word_timestamps <= 0 or not c_res.word_timestamps:
    return ()
  out: list[WordTimestamp] = []
  for i in range(c_res.num_word_timestamps):
    wt = c_res.word_timestamps[i]
    out.append(
        WordTimestamp(
            word=(wt.word.decode('utf-8', errors='replace') if wt.word else ''),
            start_char=int(wt.start_char),
            end_char=int(wt.end_char),
            start_sec=round(float(wt.start_sec), 4),
            end_sec=round(float(wt.end_sec), 4),
        )
    )
  return tuple(out)


def compute_word_timestamps(
    text: str,
    num_samples: int,
    sample_rate_hz: int,
    time_offset_sec: float = 0.0,
    samples: np.ndarray | None = None,
) -> tuple[WordTimestamp, ...]:
  """Computes prosody- and waveform-aligned word timestamps via the C API."""
  if not text or num_samples <= 0 or sample_rate_hz <= 0:
    return ()
  lib = _get_lib()
  c_res = _MpTextToSpeechAudioResultC()
  err = ctypes.c_char_p()
  samples_ptr = None
  effective_num_samples = int(num_samples)
  arr_f32 = None
  if samples is not None and samples.size > 0:
    arr_f32 = np.ascontiguousarray(samples, dtype=np.float32)
    samples_ptr = arr_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    effective_num_samples = int(arr_f32.size)
  status = lib.MpTextToSpeechComputeWordTimestamps(
      text.encode('utf-8'),
      samples_ptr,
      effective_num_samples,
      int(sample_rate_hz),
      float(time_offset_sec),
      ctypes.byref(c_res),
      ctypes.byref(err),
  )
  _check_status(lib, status, err)
  try:
    return _extract_word_timestamps_from_c(c_res)
  finally:
    lib.MpTextToSpeechFreeAudioResult(ctypes.byref(c_res))


@dataclasses.dataclass(frozen=True)
class TtsAudioResult:
  """Synthesized audio chunk or full clip returned by `TextToSpeech`.

  Attributes:
    samples: 1D NumPy array of mono float32 PCM samples in `[-1.0, 1.0]`.
    sample_rate_hz: Output PCM sample rate in Hz.
    chunk_index: Zero-based clause/chunk index in a streaming session.
    is_final: Whether this chunk is the final output in the stream.
    word_timestamps: Word-level timing alignments for read-along highlighting.
  """

  samples: np.ndarray
  sample_rate_hz: int
  chunk_index: int = 0
  is_final: bool = True
  word_timestamps: tuple[WordTimestamp, ...] = ()

  @property
  def duration_seconds(self) -> float:
    """Returns the duration of `samples` in seconds."""
    return (
        float(self.samples.size) / float(self.sample_rate_hz)
        if self.sample_rate_hz > 0
        else 0.0
    )


@dataclasses.dataclass
class TextToSpeechOptions:
  """Options for MediaPipe TextToSpeech task.

  Attributes:
    base_options: Base options for the MediaPipe task (model path and delegate).
    default_voice_id: Default speaker voice ID when not overridden per call.
    default_locale: Default BCP-47 locale tag when not overridden per call.
    speaking_rate: Playback speaking rate multiplier (`1.0` is normal speed).
    target_bucket: Fixed phoneme token bucket length (`0` selects adaptive
      bucketing).
    cache_directory: Optional directory path for compiled weight/program caches;
      `None` disables disk caching.
  """

  base_options: _BaseOptions
  default_voice_id: str = ''
  default_locale: str = 'en-US'
  speaking_rate: float = 1.0
  target_bucket: int = 0
  cache_directory: str | None = None


class TextToSpeech:
  """On-device speech synthesis via MediaPipe CalculatorGraph."""

  def __init__(self, handle: ctypes.c_void_p):
    """Initializes a `TextToSpeech` instance wrapping a C API handle.

    Args:
      handle: Opaque C pointer created by `MpTextToSpeechCreate`.
    """
    self._lib = _get_lib()
    self._handle = handle

  @classmethod
  def create_from_options(cls, options: TextToSpeechOptions) -> 'TextToSpeech':
    """Creates a TextToSpeech instance from `TextToSpeechOptions`."""
    lib = _get_lib()
    c_base = options.base_options.to_ctypes()
    voice_bytes = (
        options.default_voice_id.encode('utf-8')
        if options.default_voice_id
        else None
    )
    locale_bytes = (
        options.default_locale.encode('utf-8')
        if options.default_locale
        else None
    )
    cache_bytes = (
        options.cache_directory.encode('utf-8')
        if options.cache_directory
        else None
    )
    c_opts = _MpTextToSpeechOptionsC(
        base_options=c_base,
        default_voice_id=voice_bytes,
        default_locale=locale_bytes,
        speaking_rate=float(options.speaking_rate),
        target_bucket=int(options.target_bucket),
        cache_directory=cache_bytes,
    )
    handle = ctypes.c_void_p()
    err = ctypes.c_char_p()
    status = lib.MpTextToSpeechCreate(
        ctypes.byref(c_opts), ctypes.byref(handle), ctypes.byref(err)
    )
    _check_status(lib, status, err)
    return cls(handle)

  @classmethod
  def create_from_model_path(
      cls,
      model_path: str,
      delegate: _BaseOptions.Delegate = _BaseOptions.Delegate.CPU,
      default_voice_id: str = '',
      speaking_rate: float = 1.0,
  ) -> 'TextToSpeech':
    """Creates a TextToSpeech instance from a `.litertlm` model path."""
    opts = TextToSpeechOptions(
        base_options=_BaseOptions(
            model_asset_path=model_path, delegate=delegate
        ),
        default_voice_id=default_voice_id,
        speaking_rate=speaking_rate,
    )
    return cls.create_from_options(opts)

  def list_voices(self) -> list[VoiceInfo]:
    """Lists all voices available in the loaded model bundle."""
    c_list = _MpTextToSpeechVoiceListC()
    err = ctypes.c_char_p()
    status = self._lib.MpTextToSpeechListVoices(
        self._handle, ctypes.byref(c_list), ctypes.byref(err)
    )
    _check_status(self._lib, status, err)
    voices: list[VoiceInfo] = []
    try:
      for i in range(c_list.num_voices):
        v = c_list.voices[i]
        voices.append(
            VoiceInfo(
                voice_id=(
                    v.voice_id.decode('utf-8', errors='replace')
                    if v.voice_id
                    else ''
                ),
                display_name=(
                    v.display_name.decode('utf-8', errors='replace')
                    if v.display_name
                    else ''
                ),
                locale=(
                    v.locale.decode('utf-8', errors='replace')
                    if v.locale
                    else ''
                ),
                sample_rate_hz=int(v.sample_rate_hz),
                is_downloaded=bool(v.is_downloaded),
            )
        )
    finally:
      self._lib.MpTextToSpeechVoiceListFree(ctypes.byref(c_list))
    return voices

  def synthesize(
      self,
      text: str,
      voice_id: str | None = None,
      locale: str | None = None,
  ) -> TtsAudioResult:
    """Synthesizes `text` synchronously and returns mono float32 PCM audio.

    Args:
      text: Input text to synthesize.
      voice_id: Optional speaker voice ID override; if `None`, falls back to
        `default_voice_id` configured in `TextToSpeechOptions`.
      locale: Optional BCP-47 locale override; if `None`, falls back to
        `default_locale` configured in `TextToSpeechOptions`.

    Returns:
      A `TtsAudioResult` containing the full synthesized audio waveform.
    """
    c_synth_opts = _MpTextToSpeechSynthesisOptionsC(
        voice_id=voice_id.encode('utf-8') if voice_id else None,
        locale=locale.encode('utf-8') if locale else None,
    )
    c_res = _MpTextToSpeechAudioResultC()
    err = ctypes.c_char_p()
    status = self._lib.MpTextToSpeechSynthesize(
        self._handle,
        text.encode('utf-8'),
        ctypes.byref(c_synth_opts),
        ctypes.byref(c_res),
        ctypes.byref(err),
    )
    _check_status(self._lib, status, err)
    return self._extract_audio_result(c_res)

  def start_stream(
      self,
      initial_text: str = '',
      voice_id: str | None = None,
      locale: str | None = None,
      finish_input: bool = False,
  ) -> 'TtsStream':
    """Starts an incremental or single-utterance streaming synthesis session.

    Args:
      initial_text: Optional initial text fragment to enqueue immediately.
      voice_id: Optional speaker voice ID override.
      locale: Optional BCP-47 locale override.
      finish_input: If `False`, additional tokens/words can be appended via
        `TtsStream.push_text()` followed by `TtsStream.finish_input()`.

    Returns:
      An active `TtsStream` handle supporting `push_text()`, `finish_input()`,
      and iteration over `TtsAudioResult` chunks.
    """
    c_synth_opts = _MpTextToSpeechSynthesisOptionsC(
        voice_id=voice_id.encode('utf-8') if voice_id else None,
        locale=locale.encode('utf-8') if locale else None,
    )
    stream_handle = ctypes.c_void_p()
    err = ctypes.c_char_p()
    status = self._lib.MpTextToSpeechStartStream(
        self._handle,
        initial_text.encode('utf-8') if initial_text else b'',
        ctypes.byref(c_synth_opts),
        bool(finish_input),
        ctypes.byref(stream_handle),
        ctypes.byref(err),
    )
    _check_status(self._lib, status, err)
    return TtsStream(self, stream_handle)

  def synthesize_stream(
      self,
      text: str,
      voice_id: str | None = None,
      locale: str | None = None,
  ) -> Iterator[TtsAudioResult]:
    """Streams synthesized audio chunk-by-chunk.

    Args:
      text: Input text to synthesize into clause-level audio chunks.
      voice_id: Optional speaker voice ID override; if `None`, falls back to
        `default_voice_id` configured in `TextToSpeechOptions`.
      locale: Optional BCP-47 locale override; if `None`, falls back to
        `default_locale` configured in `TextToSpeechOptions`.

    Yields:
      `TtsAudioResult` chunks as each clause finishes acoustic + vocoder
      synthesis.
    """
    with self.start_stream(
        initial_text=text,
        voice_id=voice_id,
        locale=locale,
        finish_input=True,
    ) as stream:
      yield from stream

  def _extract_audio_result(
      self,
      c_res: _MpTextToSpeechAudioResultC,
      chunk_index: int = 0,
      is_final: bool = True,
  ) -> TtsAudioResult:
    """Converts and frees a C audio result struct."""
    try:
      if c_res.num_samples > 0 and c_res.samples:
        samples = np.ctypeslib.as_array(
            c_res.samples, shape=(c_res.num_samples,)
        ).copy()
      else:
        samples = np.zeros(0, dtype=np.float32)
      sr = int(c_res.sample_rate_hz)
      word_timestamps = _extract_word_timestamps_from_c(c_res)
    finally:
      self._lib.MpTextToSpeechFreeAudioResult(ctypes.byref(c_res))
    return TtsAudioResult(
        samples=samples,
        sample_rate_hz=sr,
        chunk_index=chunk_index,
        is_final=is_final,
        word_timestamps=word_timestamps,
    )

  def cancel(self) -> None:
    """Cancels any in-flight synthesis or streaming session."""
    if self._handle:
      self._lib.MpTextToSpeechCancel(self._handle)

  def close(self) -> None:
    """Closes the synthesizer and releases native C++ resources."""
    if self._handle:
      self._lib.MpTextToSpeechClose(self._handle)
      self._handle = ctypes.c_void_p()

  def __enter__(self) -> 'TextToSpeech':
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    self.close()


class TtsStream:
  """Bidirectional streaming synthesis session supporting `push_text()`."""

  def __init__(self, tts: TextToSpeech, stream_handle: ctypes.c_void_p):
    self._tts = tts
    self._lib = tts._lib  # pylint: disable=protected-access
    self._handle = stream_handle

  def push_text(self, text_delta: str) -> None:
    """Appends an incremental text fragment to the stream."""
    if not self._handle or not text_delta:
      return
    err = ctypes.c_char_p()
    st = self._lib.MpTextToSpeechStreamPushText(
        self._handle, text_delta.encode('utf-8'), ctypes.byref(err)
    )
    _check_status(self._lib, st, err)

  def finish_input(self) -> None:
    """Marks the end of incremental text input and flushes clauses."""
    if not self._handle:
      return
    err = ctypes.c_char_p()
    st = self._lib.MpTextToSpeechStreamFinishInput(
        self._handle, ctypes.byref(err)
    )
    _check_status(self._lib, st, err)

  def cancel(self) -> None:
    """Cancels this stream immediately."""
    if self._handle:
      self._lib.MpTextToSpeechStreamCancel(self._handle)

  def close(self) -> None:
    """Releases native stream resources."""
    if self._handle:
      self._lib.MpTextToSpeechStreamClose(self._handle)
      self._handle = ctypes.c_void_p()

  def __iter__(self) -> Iterator[TtsAudioResult]:
    while self._handle:
      c_chunk = _MpTextToSpeechAudioResultC()
      chunk_idx = ctypes.c_int32(0)
      is_final = ctypes.c_bool(False)
      chunk_err = ctypes.c_char_p()
      st = self._lib.MpTextToSpeechStreamReadNext(
          self._handle,
          ctypes.byref(c_chunk),
          ctypes.byref(chunk_idx),
          ctypes.byref(is_final),
          ctypes.byref(chunk_err),
      )
      if st == _MP_OUT_OF_RANGE:
        if chunk_err.value:
          self._lib.MpTextToSpeechFreeError(chunk_err)
        break
      _check_status(self._lib, st, chunk_err)
      yield self._tts._extract_audio_result(  # pylint: disable=protected-access
          c_chunk,
          chunk_index=int(chunk_idx.value),
          is_final=bool(is_final.value),
      )

  def __enter__(self) -> 'TtsStream':
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    self.close()
