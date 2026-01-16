# Copyright 2022 The MediaPipe Authors.
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
"""MediaPipe audio classifier task."""

import ctypes
import dataclasses
from typing import Callable, Optional

from mediapipe.tasks.python.audio.core import audio_record
from mediapipe.tasks.python.audio.core import audio_task_running_mode
from mediapipe.tasks.python.components.containers import audio_data
from mediapipe.tasks.python.components.containers import audio_data_c
from mediapipe.tasks.python.components.containers import classification_result
from mediapipe.tasks.python.components.containers import classification_result_c
from mediapipe.tasks.python.components.processors import classifier_options as classifier_options_lib
from mediapipe.tasks.python.components.processors import classifier_options_c
from mediapipe.tasks.python.core import async_result_dispatcher
from mediapipe.tasks.python.core import base_options as base_options_lib
from mediapipe.tasks.python.core import base_options_c
from mediapipe.tasks.python.core import mediapipe_c_bindings
from mediapipe.tasks.python.core import mediapipe_c_utils
from mediapipe.tasks.python.core import serial_dispatcher
from mediapipe.tasks.python.core.optional_dependencies import doc_controls

AudioClassifierResult = classification_result.ClassificationResult
_AudioData = audio_data.AudioData
_BaseOptions = base_options_lib.BaseOptions
_RunningMode = audio_task_running_mode.AudioTaskRunningMode
_MICRO_SECONDS_PER_MILLISECOND = 1000
_AsyncResultDispatcher = async_result_dispatcher.AsyncResultDispatcher
_LiveStreamPacket = async_result_dispatcher.LiveStreamPacket


class AudioClassifierResultC(ctypes.Structure):
  """The C representation of a list of audio classification results."""

  _fields_ = [
      (
          'results',
          ctypes.POINTER(classification_result_c.ClassificationResultC),
      ),
      ('results_count', ctypes.c_int),
  ]

_C_TYPES_RESULT_CALLBACK = ctypes.CFUNCTYPE(
    None, ctypes.c_int32, ctypes.POINTER(AudioClassifierResultC)
)


class AudioClassifierOptionsC(ctypes.Structure):
  """The audio classifier options used in the C API."""

  _fields_ = [
      ('base_options', base_options_c.BaseOptionsC),
      ('classifier_options', classifier_options_c.ClassifierOptionsC),
      ('running_mode', ctypes.c_int),
      ('result_callback', _C_TYPES_RESULT_CALLBACK),
  ]

  @classmethod
  @doc_controls.do_not_generate_docs
  def from_c_options(
      cls,
      base_options: base_options_c.BaseOptionsC,
      classifier_options: classifier_options_c.ClassifierOptionsC,
      running_mode: _RunningMode,
      result_callback: _C_TYPES_RESULT_CALLBACK,
  ) -> 'AudioClassifierOptionsC':
    """Creates an AudioClassifierOptionsC object from the given options."""
    return cls(
        base_options=base_options,
        classifier_options=classifier_options,
        running_mode=running_mode.ctype,
        result_callback=result_callback,
    )


_CTYPES_SIGNATURES = (
    mediapipe_c_utils.CStatusFunction(
        'MpAudioClassifierCreate',
        (
            ctypes.POINTER(AudioClassifierOptionsC),
            ctypes.POINTER(ctypes.c_void_p),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpAudioClassifierClassify',
        (
            ctypes.c_void_p,
            ctypes.POINTER(audio_data_c.AudioDataC),
            ctypes.POINTER(AudioClassifierResultC),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpAudioClassifierClassifyAsync',
        (
            ctypes.c_void_p,
            ctypes.POINTER(audio_data_c.AudioDataC),
            ctypes.c_int64,
        ),
    ),
    mediapipe_c_utils.CFunction(
        'MpAudioClassifierCloseResult',
        [ctypes.POINTER(AudioClassifierResultC)],
        None,
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpAudioClassifierClose',
        (ctypes.c_void_p,),
    ),
)


@dataclasses.dataclass
class AudioClassifierOptions:
  """Options for the audio classifier task.

  Attributes:
    base_options: Base options for the audio classifier task.
    running_mode: The running mode of the task. Default to the audio clips mode.
      Audio classifier task has two running modes: 1) The audio clips mode for
      running classification on independent audio clips. 2) The audio stream
      mode for running classification on the audio stream, such as from
      microphone. In this mode,  the "result_callback" below must be specified
      to receive the classification results asynchronously.
    display_names_locale: The locale to use for display names specified through
      the TFLite Model Metadata.
    max_results: The maximum number of top-scored classification results to
      return.
    score_threshold: Overrides the ones provided in the model metadata. Results
      below this value are rejected.
    category_allowlist: Allowlist of category names. If non-empty,
      classification results whose category name is not in this set will be
      filtered out. Duplicate or unknown category names are ignored. Mutually
      exclusive with `category_denylist`.
    category_denylist: Denylist of category names. If non-empty, classification
      results whose category name is in this set will be filtered out. Duplicate
      or unknown category names are ignored. Mutually exclusive with
      `category_allowlist`.
    result_callback: The user-defined result callback for processing audio
      stream data. The result callback should only be specified when the running
      mode is set to the audio stream mode.
  """
  base_options: _BaseOptions
  running_mode: _RunningMode = _RunningMode.AUDIO_CLIPS
  display_names_locale: Optional[str] = None
  max_results: Optional[int] = None
  score_threshold: Optional[float] = None
  category_allowlist: Optional[list[str]] = None
  category_denylist: Optional[list[str]] = None
  result_callback: Optional[Callable[[AudioClassifierResult, int], None]] = None


class AudioClassifier:
  """Class that performs audio classification on audio data.

  This API expects a TFLite model with mandatory TFLite Model Metadata that
  contains the mandatory AudioProperties of the solo input audio tensor and the
  optional (but recommended) category labels as AssociatedFiles with type
  TENSOR_AXIS_LABELS per output classification tensor.

  Input tensor:
    (kTfLiteFloat32)
    - input audio buffer of size `[batch * samples]`.
    - batch inference is not supported (`batch` is required to be 1).
    - for multi-channel models, the channels must be interleaved.
  At least one output tensor with:
    (kTfLiteFloat32)
    - `[1 x N]` array with `N` represents the number of categories.
    - optional (but recommended) category labels as AssociatedFiles with type
      TENSOR_AXIS_LABELS, containing one label per line. The first such
      AssociatedFile (if any) is used to fill the `category_name` field of the
      results. The `display_name` field is filled from the AssociatedFile (if
      any) whose locale matches the `display_names_locale` field of the
      `AudioClassifierOptions` used at creation time ("en" by default, i.e.
      English). If none of these are available, only the `index` field of the
      results will be filled.
  """
  _lib: serial_dispatcher.SerialDispatcher
  _handle: ctypes.c_void_p
  _dispatcher: _AsyncResultDispatcher
  _async_callback: _C_TYPES_RESULT_CALLBACK

  def __init__(
      self,
      lib: serial_dispatcher.SerialDispatcher,
      handle: ctypes.c_void_p,
      dispatcher: _AsyncResultDispatcher,
      async_callback: _C_TYPES_RESULT_CALLBACK,
  ):
    """Initializes the AudioClassifier instance.

    Args:
      lib: The serial dispatcher for the audio classifier task.
      handle: The handle to the audio classifier task.
      dispatcher: The async result dispatcher for the audio classifier task.
      async_callback: The C callback for processing audio stream data.
    """
    self._lib = lib
    self._handle = handle
    self._dispatcher = dispatcher
    self._async_callback = async_callback

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'AudioClassifier':
    """Creates an `AudioClassifier` object from a TensorFlow Lite model and the default `AudioClassifierOptions`.

    Note that the created `AudioClassifier` instance is in audio clips mode, for
    classifying on independent audio clips.

    Args:
      model_path: Path to the model.

    Returns:
      `AudioClassifier` object that's created from the model file and the
      default `AudioClassifierOptions`.

    Raises:
      ValueError: If failed to create `AudioClassifier` object from the provided
        file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = AudioClassifierOptions(
        base_options=base_options, running_mode=_RunningMode.AUDIO_CLIPS
    )
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(
      cls, options: AudioClassifierOptions
  ) -> 'AudioClassifier':
    """Creates the `AudioClassifier` object from audio classifier options.

    Args:
      options: Options for the audio classifier task.

    Returns:
      `AudioClassifier` object that's created from `options`.

    Raises:
      ValueError: If failed to create `AudioClassifier` object from
        `AudioClassifierOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """
    lib = mediapipe_c_bindings.load_shared_library(_CTYPES_SIGNATURES)

    def convert_result(c_result_ptr: ctypes.POINTER(AudioClassifierResultC)):
      c_result = c_result_ptr[0]
      if c_result.results_count == 0:
        raise RuntimeError('No results returned from audio classifier.')
      py_result = AudioClassifierResult.from_ctypes(c_result.results[0])
      return (py_result, py_result.timestamp_ms)

    dispatcher = _AsyncResultDispatcher(converter=convert_result)

    c_callback = dispatcher.wrap_callback(
        options.result_callback, _C_TYPES_RESULT_CALLBACK
    )
    ctypes_options = AudioClassifierOptionsC.from_c_options(
        base_options=options.base_options.to_ctypes(),
        classifier_options=classifier_options_c.convert_to_classifier_options_c(
            classifier_options_lib.ClassifierOptions(
                score_threshold=options.score_threshold,
                category_allowlist=options.category_allowlist,
                category_denylist=options.category_denylist,
                display_names_locale=options.display_names_locale,
                max_results=options.max_results,
            )
        ),
        running_mode=options.running_mode,
        result_callback=c_callback,
    )

    classifier_handle_ptr = ctypes.c_void_p()
    lib.MpAudioClassifierCreate(
        ctypes.byref(ctypes_options), ctypes.byref(classifier_handle_ptr)
    )
    return AudioClassifier(
        lib=lib,
        handle=classifier_handle_ptr,
        dispatcher=dispatcher,
        async_callback=c_callback,
    )

  def classify(self, audio_clip: _AudioData) -> list[AudioClassifierResult]:
    """Performs audio classification on the provided audio clip.

    The audio clip is represented as a MediaPipe AudioData. The method accepts
    audio clips with various length and audio sample rate. It's required to
    provide the corresponding audio sample rate within the `AudioData` object.

    The input audio clip may be longer than what the model is able to process
    in a single inference. When this occurs, the input audio clip is split into
    multiple chunks starting at different timestamps. For this reason, this
    function returns a vector of ClassificationResult objects, each associated
    ith a timestamp corresponding to the start (in milliseconds) of the chunk
    data that was classified, e.g:

    ClassificationResult #0 (first chunk of data):
      timestamp_ms: 0 (starts at 0ms)
      classifications #0 (single head model):
        category #0:
          category_name: "Speech"
          score: 0.6
        category #1:
          category_name: "Music"
          score: 0.2
    ClassificationResult #1 (second chunk of data):
      timestamp_ms: 800 (starts at 800ms)
      classifications #0 (single head model):
        category #0:
          category_name: "Speech"
          score: 0.5
       category #1:
         category_name: "Silence"
         score: 0.1

    Args:
      audio_clip: MediaPipe AudioData.

    Returns:
      An `AudioClassifierResult` object that contains a list of
      classification result objects, each associated with a timestamp
      corresponding to the start (in milliseconds) of the chunk data that was
      classified.

    Raises:
      ValueError: If any of the input arguments is invalid, such as the sample
        rate is not provided in the `AudioData` object.
      RuntimeError: If audio classification failed to run.
    """
    if not audio_clip.audio_format.sample_rate:
      raise ValueError('Must provide the audio sample rate in audio data.')

    c_result = AudioClassifierResultC()
    self._lib.MpAudioClassifierClassify(
        self._handle,
        audio_clip.to_ctypes(),
        ctypes.byref(c_result),
    )
    py_result = [
        AudioClassifierResult.from_ctypes(c_result.results[i])
        for i in range(c_result.results_count)
    ]
    self._lib.MpAudioClassifierCloseResult(ctypes.byref(c_result))
    return py_result

  def classify_async(self, audio_block: _AudioData, timestamp_ms: int) -> None:
    """Sends audio data (a block in a continuous audio stream) to perform audio classification.

    Only use this method when the AudioClassifier is created with the audio
    stream running mode. The input timestamps should be monotonically increasing
    for adjacent calls of this method. This method will return immediately after
    the input audio data is accepted. The results will be available via the
    `result_callback` provided in the `AudioClassifierOptions`. The
    `classify_async` method is designed to process auido stream data such as
    microphone input.

    The input audio data may be longer than what the model is able to process
    in a single inference. When this occurs, the input audio block is split
    into multiple chunks. For this reason, the callback may be called multiple
    times (once per chunk) for each call to this function.

    The `result_callback` provides:
      - An `AudioClassifierResult` object that contains a list of
        classifications.
      - The input timestamp in milliseconds.

    Args:
      audio_block: MediaPipe AudioData.
      timestamp_ms: The timestamp of the input audio data in milliseconds.

    Raises:
      ValueError: If any of the followings:
        1) The sample rate is not provided in the `AudioData` object or the
        provided sample rate is inconsistent with the previously received.
        2) The current input timestamp is smaller than what the audio
        classifier has already processed.
    """
    if not audio_block.audio_format.sample_rate:
      raise ValueError('Must provide the audio sample rate in audio data.')

    self._lib.MpAudioClassifierClassifyAsync(
        self._handle,
        audio_block.to_ctypes(),
        timestamp_ms,
    )

  def create_audio_record(
      self, num_channels: int, sample_rate: int, required_input_buffer_size: int
  ) -> audio_record.AudioRecord:
    """Creates an AudioRecord instance to record audio stream.

    The returned AudioRecord instance is initialized and client needs to call
    the appropriate method to start recording.

    Note that MediaPipe Audio tasks will up/down sample automatically to fit the
    sample rate required by the model. The default sample rate of the MediaPipe
    pretrained audio model, Yamnet is 16kHz.

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
      self._lib.MpAudioClassifierClose(self._handle)
      self._handle = None
      self._dispatcher.close()
      self._lib.close()

  def __enter__(self):
    """Returns `self` upon entering the runtime context."""
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Shuts down the MediaPipe task instance on exit of the context manager.

    Args:
      exc_type: The exception type that caused the exit.
      exc_value: The exception value that caused the exit.
      traceback: The exception traceback that caused the exit.

    Raises:
      RuntimeError: If the MediaPipe TextClassifier task failed to close.
    """
    del exc_type, exc_value, traceback  # Unused.
    self.close()

  def __del__(self):
    self.close()
