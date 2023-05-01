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
"""MediaPipe vision task base api."""

import math
from typing import Callable, Mapping, Optional

from mediapipe.framework import calculator_pb2
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.python._framework_bindings import packet as packet_module
from mediapipe.python._framework_bindings import task_runner as task_runner_module
from mediapipe.tasks.python.components.containers import rect as rect_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

_TaskRunner = task_runner_module.TaskRunner
_Packet = packet_module.Packet
_NormalizedRect = rect_module.NormalizedRect
_RunningMode = running_mode_module.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions


@doc_controls.do_not_generate_docs
class BaseVisionTaskApi(object):
  """The base class of the user-facing mediapipe vision task api classes."""

  def __init__(
      self,
      graph_config: calculator_pb2.CalculatorGraphConfig,
      running_mode: _RunningMode,
      packet_callback: Optional[
          Callable[[Mapping[str, packet_module.Packet]], None]
      ] = None,
  ) -> None:
    """Initializes the `BaseVisionTaskApi` object.

    Args:
      graph_config: The mediapipe vision task graph config proto.
      running_mode: The running mode of the mediapipe vision task.
      packet_callback: The optional packet callback for getting results
        asynchronously in the live stream mode.

    Raises:
      ValueError: The packet callback is not properly set based on the task's
      running mode.
    """
    if running_mode == _RunningMode.LIVE_STREAM:
      if packet_callback is None:
        raise ValueError(
            'The vision task is in live stream mode, a user-defined result '
            'callback must be provided.'
        )
    elif packet_callback:
      raise ValueError(
          'The vision task is in image or video mode, a user-defined result '
          'callback should not be provided.'
      )
    self._runner = _TaskRunner.create(graph_config, packet_callback)
    self._running_mode = running_mode

  def _process_image_data(
      self, inputs: Mapping[str, _Packet]
  ) -> Mapping[str, _Packet]:
    """A synchronous method to process single image inputs.

    The call blocks the current thread until a failure status or a successful
    result is returned.

    Args:
      inputs: A dict contains (input stream name, data packet) pairs.

    Returns:
      A dict contains (output stream name, data packet) pairs.

    Raises:
      ValueError: If the task's running mode is not set to image mode.
    """
    if self._running_mode != _RunningMode.IMAGE:
      raise ValueError(
          'Task is not initialized with the image mode. Current running mode:'
          + self._running_mode.name
      )
    return self._runner.process(inputs)

  def _process_video_data(
      self, inputs: Mapping[str, _Packet]
  ) -> Mapping[str, _Packet]:
    """A synchronous method to process continuous video frames.

    The call blocks the current thread until a failure status or a successful
    result is returned.

    Args:
      inputs: A dict contains (input stream name, data packet) pairs.

    Returns:
      A dict contains (output stream name, data packet) pairs.

    Raises:
      ValueError: If the task's running mode is not set to the video mode.
    """
    if self._running_mode != _RunningMode.VIDEO:
      raise ValueError(
          'Task is not initialized with the video mode. Current running mode:'
          + self._running_mode.name
      )
    return self._runner.process(inputs)

  def _send_live_stream_data(self, inputs: Mapping[str, _Packet]) -> None:
    """An asynchronous method to send live stream data to the runner.

    The results will be available in the user-defined results callback.

    Args:
      inputs: A dict contains (input stream name, data packet) pairs.

    Raises:
      ValueError: If the task's running mode is not set to the live stream
      mode.
    """
    if self._running_mode != _RunningMode.LIVE_STREAM:
      raise ValueError(
          'Task is not initialized with the live stream mode. Current running'
          ' mode:'
          + self._running_mode.name
      )
    self._runner.send(inputs)

  def convert_to_normalized_rect(
      self,
      options: _ImageProcessingOptions,
      image: image_module.Image,
      roi_allowed: bool = True,
  ) -> _NormalizedRect:
    """Converts from ImageProcessingOptions to NormalizedRect, performing sanity checks on-the-fly.

    If the input ImageProcessingOptions is not present, returns a default
    NormalizedRect covering the whole image with rotation set to 0. If
    'roi_allowed' is false, an error will be returned if the input
    ImageProcessingOptions has its 'region_of_interest' field set.

    Args:
      options: Options for image processing.
      image: The image to process.
      roi_allowed: Indicates if the `region_of_interest` field is allowed to be
        set. By default, it's set to True.

    Returns:
      A normalized rect proto that represents the image processing options.
    """
    normalized_rect = _NormalizedRect(
        rotation=0, x_center=0.5, y_center=0.5, width=1, height=1
    )
    if options is None:
      return normalized_rect

    if options.rotation_degrees % 90 != 0:
      raise ValueError('Expected rotation to be a multiple of 90°.')

    # Convert to radians counter-clockwise.
    normalized_rect.rotation = -options.rotation_degrees * math.pi / 180.0

    if options.region_of_interest:
      if not roi_allowed:
        raise ValueError("This task doesn't support region-of-interest.")
      roi = options.region_of_interest
      if roi.left >= roi.right or roi.top >= roi.bottom:
        raise ValueError('Expected Rect with left < right and top < bottom.')
      if roi.left < 0 or roi.top < 0 or roi.right > 1 or roi.bottom > 1:
        raise ValueError('Expected Rect values to be in [0,1].')
      normalized_rect.x_center = (roi.left + roi.right) / 2.0
      normalized_rect.y_center = (roi.top + roi.bottom) / 2.0
      normalized_rect.width = roi.right - roi.left
      normalized_rect.height = roi.bottom - roi.top

    # For 90° and 270° rotations, we need to swap width and height.
    # This is due to the internal behavior of ImageToTensorCalculator, which:
    # - first denormalizes the provided rect by multiplying the rect width or
    #   height by the image width or height, respectively.
    # - then rotates this by denormalized rect by the provided rotation, and
    #   uses this for cropping,
    # - then finally rotates this back.
    if abs(options.rotation_degrees % 180) != 0:
      w = normalized_rect.height * image.height / image.width
      h = normalized_rect.width * image.width / image.height
      normalized_rect.width = w
      normalized_rect.height = h

    return normalized_rect

  def close(self) -> None:
    """Shuts down the mediapipe vision task instance.

    Raises:
      RuntimeError: If the mediapipe vision task failed to close.
    """
    self._runner.close()

  def get_graph_config(self) -> calculator_pb2.CalculatorGraphConfig:
    """Returns the canonicalized CalculatorGraphConfig of the underlying graph.
    """
    return self._runner.get_graph_config()

  def __enter__(self):
    """Return `self` upon entering the runtime context."""
    return self

  def __exit__(self, unused_exc_type, unused_exc_value, unused_traceback):
    """Shuts down the mediapipe vision task instance on exit of the context manager.

    Raises:
      RuntimeError: If the mediapipe vision task failed to close.
    """
    self.close()
