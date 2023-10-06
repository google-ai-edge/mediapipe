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

import dataclasses
from typing import Callable, Mapping, List, Optional

from mediapipe.python import packet_creator
from mediapipe.python import packet_getter
from mediapipe.python._framework_bindings import packet
from mediapipe.tasks.cc.audio.audio_classifier.proto import audio_classifier_graph_options_pb2
from mediapipe.tasks.cc.components.containers.proto import classifications_pb2
from mediapipe.tasks.cc.components.processors.proto import classifier_options_pb2
from mediapipe.tasks.python.audio.core import audio_task_running_mode as running_mode_module
from mediapipe.tasks.python.audio.core import base_audio_task_api
from mediapipe.tasks.python.components.containers import audio_data as audio_data_module
from mediapipe.tasks.python.components.containers import classification_result as classification_result_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import task_info as task_info_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls

AudioClassifierResult = classification_result_module.ClassificationResult
_AudioClassifierGraphOptionsProto = audio_classifier_graph_options_pb2.AudioClassifierGraphOptions
_AudioData = audio_data_module.AudioData
_BaseOptions = base_options_module.BaseOptions
_ClassifierOptionsProto = classifier_options_pb2.ClassifierOptions
_RunningMode = running_mode_module.AudioTaskRunningMode
_TaskInfo = task_info_module.TaskInfo

_AUDIO_IN_STREAM_NAME = 'audio_in'
_AUDIO_TAG = 'AUDIO'
_CLASSIFICATIONS_STREAM_NAME = 'classifications_out'
_CLASSIFICATIONS_TAG = 'CLASSIFICATIONS'
_SAMPLE_RATE_IN_STREAM_NAME = 'sample_rate_in'
_SAMPLE_RATE_TAG = 'SAMPLE_RATE'
_TASK_GRAPH_NAME = 'mediapipe.tasks.audio.audio_classifier.AudioClassifierGraph'
_TIMESTAMPED_CLASSIFICATIONS_STREAM_NAME = 'timestamped_classifications_out'
_TIMESTAMPED_CLASSIFICATIONS_TAG = 'TIMESTAMPED_CLASSIFICATIONS'
_MICRO_SECONDS_PER_MILLISECOND = 1000


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
  category_allowlist: Optional[List[str]] = None
  category_denylist: Optional[List[str]] = None
  result_callback: Optional[Callable[[AudioClassifierResult, int], None]] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _AudioClassifierGraphOptionsProto:
    """Generates an AudioClassifierOptions protobuf object."""
    base_options_proto = self.base_options.to_pb2()
    base_options_proto.use_stream_mode = False if self.running_mode == _RunningMode.AUDIO_CLIPS else True
    classifier_options_proto = _ClassifierOptionsProto(
        score_threshold=self.score_threshold,
        category_allowlist=self.category_allowlist,
        category_denylist=self.category_denylist,
        display_names_locale=self.display_names_locale,
        max_results=self.max_results)

    return _AudioClassifierGraphOptionsProto(
        base_options=base_options_proto,
        classifier_options=classifier_options_proto)


class AudioClassifier(base_audio_task_api.BaseAudioTaskApi):
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
        base_options=base_options, running_mode=_RunningMode.AUDIO_CLIPS)
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(cls,
                          options: AudioClassifierOptions) -> 'AudioClassifier':
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

    def packets_callback(output_packets: Mapping[str, packet.Packet]):
      timestamp_ms = output_packets[
          _CLASSIFICATIONS_STREAM_NAME].timestamp.value // _MICRO_SECONDS_PER_MILLISECOND
      if output_packets[_CLASSIFICATIONS_STREAM_NAME].is_empty():
        options.result_callback(
            AudioClassifierResult(classifications=[]), timestamp_ms)
        return
      classification_result_proto = classifications_pb2.ClassificationResult()
      classification_result_proto.CopyFrom(
          packet_getter.get_proto(output_packets[_CLASSIFICATIONS_STREAM_NAME]))
      options.result_callback(
          AudioClassifierResult.create_from_pb2(classification_result_proto),
          timestamp_ms)

    task_info = _TaskInfo(
        task_graph=_TASK_GRAPH_NAME,
        input_streams=[
            ':'.join([_AUDIO_TAG, _AUDIO_IN_STREAM_NAME]),
            ':'.join([_SAMPLE_RATE_TAG, _SAMPLE_RATE_IN_STREAM_NAME])
        ],
        output_streams=[
            ':'.join([_CLASSIFICATIONS_TAG, _CLASSIFICATIONS_STREAM_NAME]),
            ':'.join([
                _TIMESTAMPED_CLASSIFICATIONS_TAG,
                _TIMESTAMPED_CLASSIFICATIONS_STREAM_NAME
            ])
        ],
        task_options=options)
    return cls(
        # Audio tasks should not drop input audio due to flow limiting, which
        # may cause data inconsistency.
        task_info.generate_graph_config(enable_flow_limiting=False),
        options.running_mode,
        packets_callback if options.result_callback else None)

  def classify(self, audio_clip: _AudioData) -> List[AudioClassifierResult]:
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
    output_packets = self._process_audio_clip({
        _AUDIO_IN_STREAM_NAME:
            packet_creator.create_matrix(audio_clip.buffer, transpose=True),
        _SAMPLE_RATE_IN_STREAM_NAME:
            packet_creator.create_double(audio_clip.audio_format.sample_rate)
    })
    output_list = []
    classification_result_proto_list = packet_getter.get_proto_list(
        output_packets[_TIMESTAMPED_CLASSIFICATIONS_STREAM_NAME])
    for proto in classification_result_proto_list:
      classification_result_proto = classifications_pb2.ClassificationResult()
      classification_result_proto.CopyFrom(proto)
      output_list.append(
          AudioClassifierResult.create_from_pb2(classification_result_proto))
    return output_list

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
    if not self._default_sample_rate:
      self._default_sample_rate = audio_block.audio_format.sample_rate
      self._set_sample_rate(_SAMPLE_RATE_IN_STREAM_NAME,
                            self._default_sample_rate)
    elif audio_block.audio_format.sample_rate != self._default_sample_rate:
      raise ValueError(
          f'The audio sample rate provided in audio data: '
          f'{audio_block.audio_format.sample_rate} is inconsistent with '
          f'the previously received: {self._default_sample_rate}.')

    self._send_audio_stream_data({
        _AUDIO_IN_STREAM_NAME:
            packet_creator.create_matrix(audio_block.buffer, transpose=True).at(
                timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND)
    })
