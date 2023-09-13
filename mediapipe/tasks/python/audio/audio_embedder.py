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
"""MediaPipe audio embedder task."""

import dataclasses
from typing import Callable, Mapping, List, Optional

from mediapipe.python import packet_creator
from mediapipe.python import packet_getter
from mediapipe.python._framework_bindings import packet
from mediapipe.tasks.cc.audio.audio_embedder.proto import audio_embedder_graph_options_pb2
from mediapipe.tasks.cc.components.containers.proto import embeddings_pb2
from mediapipe.tasks.cc.components.processors.proto import embedder_options_pb2
from mediapipe.tasks.python.audio.core import audio_task_running_mode as running_mode_module
from mediapipe.tasks.python.audio.core import base_audio_task_api
from mediapipe.tasks.python.components.containers import audio_data as audio_data_module
from mediapipe.tasks.python.components.containers import embedding_result as embedding_result_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import task_info as task_info_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls

AudioEmbedderResult = embedding_result_module.EmbeddingResult
_AudioEmbedderGraphOptionsProto = audio_embedder_graph_options_pb2.AudioEmbedderGraphOptions
_AudioData = audio_data_module.AudioData
_BaseOptions = base_options_module.BaseOptions
_EmbedderOptionsProto = embedder_options_pb2.EmbedderOptions
_RunningMode = running_mode_module.AudioTaskRunningMode
_TaskInfo = task_info_module.TaskInfo

_AUDIO_IN_STREAM_NAME = 'audio_in'
_AUDIO_TAG = 'AUDIO'
_EMBEDDINGS_STREAM_NAME = 'embeddings_out'
_EMBEDDINGS_TAG = 'EMBEDDINGS'
_SAMPLE_RATE_IN_STREAM_NAME = 'sample_rate_in'
_SAMPLE_RATE_TAG = 'SAMPLE_RATE'
_TASK_GRAPH_NAME = 'mediapipe.tasks.audio.audio_embedder.AudioEmbedderGraph'
_TIMESTAMPTED_EMBEDDINGS_STREAM_NAME = 'timestamped_embeddings_out'
_TIMESTAMPTED_EMBEDDINGS_TAG = 'TIMESTAMPED_EMBEDDINGS'
_MICRO_SECONDS_PER_MILLISECOND = 1000


@dataclasses.dataclass
class AudioEmbedderOptions:
  """Options for the audio embedder task.

  Attributes:
    base_options: Base options for the audio embedder task.
    running_mode: The running mode of the task. Default to the audio clips mode.
      Audio embedder task has two running modes: 1) The audio clips mode for
      running embedding extraction on independent audio clips. 2) The audio
      stream mode for running embedding extraction on the audio stream, such as
      from microphone. In this mode,  the "result_callback" below must be
      specified to receive the embedding results asynchronously.
    l2_normalize: Whether to normalize the returned feature vector with L2 norm.
      Use this option only if the model does not already contain a native
      L2_NORMALIZATION TF Lite Op. In most cases, this is already the case and
      L2 norm is thus achieved through TF Lite inference.
    quantize: Whether the returned embedding should be quantized to bytes via
      scalar quantization. Embeddings are implicitly assumed to be unit-norm and
      therefore any dimension is guaranteed to have a value in [-1.0, 1.0]. Use
      the l2_normalize option if this is not the case.
    result_callback: The user-defined result callback for processing audio
      stream data. The result callback should only be specified when the running
      mode is set to the audio stream mode.
  """
  base_options: _BaseOptions
  running_mode: _RunningMode = _RunningMode.AUDIO_CLIPS
  l2_normalize: Optional[bool] = None
  quantize: Optional[bool] = None
  result_callback: Optional[Callable[[AudioEmbedderResult, int], None]] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _AudioEmbedderGraphOptionsProto:
    """Generates an AudioEmbedderOptions protobuf object."""
    base_options_proto = self.base_options.to_pb2()
    base_options_proto.use_stream_mode = False if self.running_mode == _RunningMode.AUDIO_CLIPS else True
    embedder_options_proto = _EmbedderOptionsProto(
        l2_normalize=self.l2_normalize, quantize=self.quantize)

    return _AudioEmbedderGraphOptionsProto(
        base_options=base_options_proto,
        embedder_options=embedder_options_proto)


class AudioEmbedder(base_audio_task_api.BaseAudioTaskApi):
  """Class that performs embedding extraction on audio clips or audio stream.

  This API expects a TFLite model with mandatory TFLite Model Metadata that
  contains the mandatory AudioProperties of the solo input audio tensor and the
  optional (but recommended) label items as AssociatedFiles with type
  TENSOR_AXIS_LABELS per output embedding tensor.

  Input tensor:
    (kTfLiteFloat32)
    - input audio buffer of size `[batch * samples]`.
    - batch inference is not supported (`batch` is required to be 1).
    - for multi-channel models, the channels must be interleaved.
  At least one output tensor with:
    (kTfLiteUInt8/kTfLiteFloat32)
    - `N` components corresponding to the `N` dimensions of the returned
    feature vector for this output layer.
    - Either 2 or 4 dimensions, i.e. `[1 x N]` or `[1 x 1 x 1 x N]`.
  """

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'AudioEmbedder':
    """Creates an `AudioEmbedder` object from a TensorFlow Lite model and the default `AudioEmbedderOptions`.

    Note that the created `AudioEmbedder` instance is in audio clips mode, for
    embedding extraction on the independent audio clips.

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
    options = AudioEmbedderOptions(
        base_options=base_options, running_mode=_RunningMode.AUDIO_CLIPS)
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(cls,
                          options: AudioEmbedderOptions) -> 'AudioEmbedder':
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

    def packets_callback(output_packets: Mapping[str, packet.Packet]):
      timestamp_ms = output_packets[
          _EMBEDDINGS_STREAM_NAME].timestamp.value // _MICRO_SECONDS_PER_MILLISECOND
      if output_packets[_EMBEDDINGS_STREAM_NAME].is_empty():
        options.result_callback(
            AudioEmbedderResult(embeddings=[]), timestamp_ms)
        return
      embedding_result_proto = embeddings_pb2.EmbeddingResult()
      embedding_result_proto.CopyFrom(
          packet_getter.get_proto(output_packets[_EMBEDDINGS_STREAM_NAME]))
      options.result_callback(
          AudioEmbedderResult.create_from_pb2(embedding_result_proto),
          timestamp_ms)

    task_info = _TaskInfo(
        task_graph=_TASK_GRAPH_NAME,
        input_streams=[
            ':'.join([_AUDIO_TAG, _AUDIO_IN_STREAM_NAME]),
            ':'.join([_SAMPLE_RATE_TAG, _SAMPLE_RATE_IN_STREAM_NAME])
        ],
        output_streams=[
            ':'.join([_EMBEDDINGS_TAG, _EMBEDDINGS_STREAM_NAME]), ':'.join([
                _TIMESTAMPTED_EMBEDDINGS_TAG,
                _TIMESTAMPTED_EMBEDDINGS_STREAM_NAME
            ])
        ],
        task_options=options)
    return cls(
        # Audio tasks should not drop input audio due to flow limiting, which
        # may cause data inconsistency.
        task_info.generate_graph_config(enable_flow_limiting=False),
        options.running_mode,
        packets_callback if options.result_callback else None)

  def embed(self, audio_clip: _AudioData) -> List[AudioEmbedderResult]:
    """Performs embedding extraction on the provided audio clips.

    The audio clip is represented as a MediaPipe AudioData. The method accepts
    audio clips with various length and audio sample rate. It's required to
    provide the corresponding audio sample rate within the `AudioData` object.

    The input audio clip may be longer than what the model is able to process
    in a single inference. When this occurs, the input audio clip is split into
    multiple chunks starting at different timestamps. For this reason, this
    function returns a vector of EmbeddingResult objects, each associated
    ith a timestamp corresponding to the start (in milliseconds) of the chunk
    data on which embedding extraction was carried out.

    Args:
      audio_clip: MediaPipe AudioData.

    Returns:
      An `AudioEmbedderResult` object that contains a list of embedding result
      objects, each associated with a timestamp corresponding to the start
      (in milliseconds) of the chunk data on which embedding extraction was
      carried out.

    Raises:
      ValueError: If any of the input arguments is invalid, such as the sample
        rate is not provided in the `AudioData` object.
      RuntimeError: If audio embedding extraction failed to run.
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
    embeddings_proto_list = packet_getter.get_proto_list(
        output_packets[_TIMESTAMPTED_EMBEDDINGS_STREAM_NAME])
    for proto in embeddings_proto_list:
      embedding_result_proto = embeddings_pb2.EmbeddingResult()
      embedding_result_proto.CopyFrom(proto)
      output_list.append(
          AudioEmbedderResult.create_from_pb2(embedding_result_proto))
    return output_list

  def embed_async(self, audio_block: _AudioData, timestamp_ms: int) -> None:
    """Sends audio data (a block in a continuous audio stream) to perform audio embedding extraction.

    Only use this method when the AudioEmbedder is created with the audio
    stream running mode. The input timestamps should be monotonically increasing
    for adjacent calls of this method. This method will return immediately after
    the input audio data is accepted. The results will be available via the
    `result_callback` provided in the `AudioEmbedderOptions`. The
    `embed_async` method is designed to process auido stream data such as
    microphone input.

    The input audio data may be longer than what the model is able to process
    in a single inference. When this occurs, the input audio block is split
    into multiple chunks. For this reason, the callback may be called multiple
    times (once per chunk) for each call to this function.

    The `result_callback` provides:
      - An `AudioEmbedderResult` object that contains a list of
        embeddings.
      - The input timestamp in milliseconds.

    Args:
      audio_block: MediaPipe AudioData.
      timestamp_ms: The timestamp of the input audio data in milliseconds.

    Raises:
      ValueError: If any of the followings:
        1) The sample rate is not provided in the `AudioData` object or the
        provided sample rate is inconsistent with the previously received.
        2) The current input timestamp is smaller than what the audio
        embedder has already processed.
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
