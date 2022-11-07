# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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
"""MediaPipe audio task base api."""

from typing import Callable, Mapping, Optional

from mediapipe.framework import calculator_pb2
from mediapipe.python import packet_creator
from mediapipe.python._framework_bindings import packet as packet_module
from mediapipe.python._framework_bindings import task_runner as task_runner_module
from mediapipe.python._framework_bindings import timestamp as timestamp_module
from mediapipe.tasks.python.audio.core import audio_task_running_mode as running_mode_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls

_TaskRunner = task_runner_module.TaskRunner
_Packet = packet_module.Packet
_RunningMode = running_mode_module.AudioTaskRunningMode
_Timestamp = timestamp_module.Timestamp


class BaseAudioTaskApi(object):
  """The base class of the user-facing mediapipe audio task api classes."""

  def __init__(
      self,
      graph_config: calculator_pb2.CalculatorGraphConfig,
      running_mode: _RunningMode,
      packet_callback: Optional[Callable[[Mapping[str, packet_module.Packet]],
                                         None]] = None
  ) -> None:
    """Initializes the `BaseAudioTaskApi` object.

    Args:
      graph_config: The mediapipe audio task graph config proto.
      running_mode: The running mode of the mediapipe audio task.
      packet_callback: The optional packet callback for getting results
        asynchronously in the audio stream mode.

    Raises:
      ValueError: The packet callback is not properly set based on the task's
      running mode.
    """
    if running_mode == _RunningMode.AUDIO_STREAM:
      if packet_callback is None:
        raise ValueError(
            'The audio task is in audio stream mode, a user-defined result '
            'callback must be provided.')
    elif packet_callback:
      raise ValueError(
          'The audio task is in audio clips mode, a user-defined result '
          'callback should not be provided.')
    self._runner = _TaskRunner.create(graph_config, packet_callback)
    self._running_mode = running_mode
    self._default_sample_rate = None

  def _process_audio_clip(
      self, inputs: Mapping[str, _Packet]) -> Mapping[str, _Packet]:
    """A synchronous method to process independent audio clips.

    The call blocks the current thread until a failure status or a successful
    result is returned.

    Args:
      inputs: A dict contains (input stream name, data packet) pairs.

    Returns:
      A dict contains (output stream name, data packet) pairs.

    Raises:
      ValueError: If the task's running mode is not set to audio clips mode.
    """
    if self._running_mode != _RunningMode.AUDIO_CLIPS:
      raise ValueError(
          'Task is not initialized with the audio clips mode. Current running mode:'
          + self._running_mode.name)
    return self._runner.process(inputs)

  def _set_sample_rate(self, sample_rate_stream_name: str,
                       sample_rate: float) -> None:
    """An asynchronous method to set audio sample rate in the audio stream mode.

    Args:
      sample_rate_stream_name: The audio sample rate stream name.
      sample_rate: The audio sample rate.

    Raises:
      ValueError: If the task's running mode is not set to the audio stream
      mode.
    """
    if self._running_mode != _RunningMode.AUDIO_STREAM:
      raise ValueError(
          'Task is not initialized with the audio stream mode. Current running mode:'
          + self._running_mode.name)
    self._runner.send({
        sample_rate_stream_name:
            packet_creator.create_double(sample_rate).at(_Timestamp.PRESTREAM)
    })

  def _send_audio_stream_data(self, inputs: Mapping[str, _Packet]) -> None:
    """An asynchronous method to send audio stream data to the runner.

    The results will be available in the user-defined results callback.

    Args:
      inputs: A dict contains (input stream name, data packet) pairs.

    Raises:
      ValueError: If the task's running mode is not set to the audio stream
      mode.
    """
    if self._running_mode != _RunningMode.AUDIO_STREAM:
      raise ValueError(
          'Task is not initialized with the audio stream mode. Current running mode:'
          + self._running_mode.name)
    self._runner.send(inputs)

  def close(self) -> None:
    """Shuts down the mediapipe audio task instance.

    Raises:
      RuntimeError: If the mediapipe audio task failed to close.
    """
    self._runner.close()

  @doc_controls.do_not_generate_docs
  def __enter__(self):
    """Return `self` upon entering the runtime context."""
    return self

  @doc_controls.do_not_generate_docs
  def __exit__(self, unused_exc_type, unused_exc_value, unused_traceback):
    """Shuts down the mediapipe audio task instance on exit of the context manager.

    Raises:
      RuntimeError: If the mediapipe audio task failed to close.
    """
    self.close()
