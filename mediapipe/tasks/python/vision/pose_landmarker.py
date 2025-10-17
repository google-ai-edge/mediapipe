# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
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
"""MediaPipe pose landmarker task."""

import ctypes
import dataclasses
import logging
from typing import Callable, Optional

from mediapipe.tasks.python.components.containers import landmark as landmark_lib
from mediapipe.tasks.python.components.containers import landmark_c as landmark_c_lib
from mediapipe.tasks.python.core import base_options as base_options_lib
from mediapipe.tasks.python.core import base_options_c as base_options_c_lib
from mediapipe.tasks.python.core import mediapipe_c_bindings as mediapipe_c_bindings_lib
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import base_vision_task_api
from mediapipe.tasks.python.vision.core import image as image_lib
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_lib
from mediapipe.tasks.python.vision.core import image_processing_options_c as image_processing_options_c_lib
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_lib

_BaseOptions = base_options_lib.BaseOptions
_RunningMode = running_mode_lib.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_lib.ImageProcessingOptions


class PoseLandmarkerResultC(ctypes.Structure):
  """The ctypes struct for PoseLandmarkerResult."""

  _fields_ = [
      ('segmentation_masks', ctypes.POINTER(ctypes.c_void_p)),
      ('segmentation_masks_count', ctypes.c_uint32),
      ('pose_landmarks', ctypes.POINTER(landmark_c_lib.NormalizedLandmarksC)),
      ('pose_landmarks_count', ctypes.c_uint32),
      ('pose_world_landmarks', ctypes.POINTER(landmark_c_lib.LandmarksC)),
      ('pose_world_landmarks_count', ctypes.c_uint32),
  ]


@dataclasses.dataclass
class PoseLandmarkerResult:
  """The pose landmarks detection result from PoseLandmarker, where each vector element represents a single pose detected in the image.

  Attributes:
    pose_landmarks: Detected pose landmarks in normalized image coordinates.
    pose_world_landmarks:  Detected pose landmarks in world coordinates.
    segmentation_masks: Optional segmentation masks for pose.
  """

  pose_landmarks: list[list[landmark_lib.NormalizedLandmark]]
  pose_world_landmarks: list[list[landmark_lib.Landmark]]
  segmentation_masks: Optional[list[image_lib.Image]] = None

  @classmethod
  @doc_controls.do_not_generate_docs
  def from_ctypes(
      cls, c_struct: PoseLandmarkerResultC
  ) -> 'PoseLandmarkerResult':
    """Creates a PoseLandmarkerResult from a ctypes struct."""
    lib = mediapipe_c_bindings_lib.load_shared_library()
    pose_landmarks = []
    for i in range(c_struct.pose_landmarks_count):
      landmarks_c = c_struct.pose_landmarks[i]
      pose_landmarks.append([
          landmark_lib.NormalizedLandmark.from_ctypes(landmarks_c.landmarks[j])
          for j in range(landmarks_c.landmarks_count)
      ])

    pose_world_landmarks = []
    for i in range(c_struct.pose_world_landmarks_count):
      landmarks_c = c_struct.pose_world_landmarks[i]
      pose_world_landmarks.append([
          landmark_lib.Landmark.from_ctypes(landmarks_c.landmarks[j])
          for j in range(landmarks_c.landmarks_count)
      ])

    segmentation_masks = []
    for i in range(c_struct.segmentation_masks_count):
      image_c = c_struct.segmentation_masks[i]
      segmentation_masks.append(
          image_lib.Image.create_from_ctypes(image_c, lib)
      )

    segmentation_masks = (
        None if c_struct.segmentation_masks_count == 0 else segmentation_masks
    )
    return cls(pose_landmarks, pose_world_landmarks, segmentation_masks)


class PoseLandmarksConnections:
  """The connections between pose landmarks."""

  @dataclasses.dataclass
  class Connection:
    """The connection class for pose landmarks."""

    start: int
    end: int

  POSE_LANDMARKS: list[Connection] = [
      Connection(0, 1),
      Connection(1, 2),
      Connection(2, 3),
      Connection(3, 7),
      Connection(0, 4),
      Connection(4, 5),
      Connection(5, 6),
      Connection(6, 8),
      Connection(9, 10),
      Connection(11, 12),
      Connection(11, 13),
      Connection(13, 15),
      Connection(15, 17),
      Connection(15, 19),
      Connection(15, 21),
      Connection(17, 19),
      Connection(12, 14),
      Connection(14, 16),
      Connection(16, 18),
      Connection(16, 20),
      Connection(16, 22),
      Connection(18, 20),
      Connection(11, 23),
      Connection(12, 24),
      Connection(23, 24),
      Connection(23, 25),
      Connection(24, 26),
      Connection(25, 27),
      Connection(26, 28),
      Connection(27, 29),
      Connection(28, 30),
      Connection(29, 31),
      Connection(30, 32),
      Connection(27, 31),
      Connection(28, 32),
  ]


class PoseLandmarkerOptionsC(ctypes.Structure):
  """The ctypes struct for PoseLandmarkerOptions."""

  _fields_ = [
      ('base_options', base_options_c_lib.BaseOptionsC),
      ('running_mode', ctypes.c_int),
      ('num_poses', ctypes.c_int),
      ('min_pose_detection_confidence', ctypes.c_float),
      ('min_pose_presence_confidence', ctypes.c_float),
      ('min_tracking_confidence', ctypes.c_float),
      ('output_segmentation_masks', ctypes.c_bool),
      (
          'result_callback',
          ctypes.CFUNCTYPE(
              None,
              ctypes.POINTER(PoseLandmarkerResultC),
              ctypes.c_void_p,
              ctypes.c_int64,
              ctypes.c_char_p,
          ),
      ),
  ]


@dataclasses.dataclass
class PoseLandmarkerOptions:
  """Options for the pose landmarker task.

  Attributes:
    base_options: Base options for the pose landmarker task.
    running_mode: The running mode of the task. Default to the image mode.
      PoseLandmarker has three running modes: 1) The image mode for detecting
      pose landmarks on single image inputs. 2) The video mode for detecting
      pose landmarks on the decoded frames of a video. 3) The live stream mode
      for detecting pose landmarks on the live stream of input data, such as
      from camera. In this mode, the "result_callback" below must be specified
      to receive the detection results asynchronously.
    num_poses: The maximum number of poses can be detected by the
      PoseLandmarker.
    min_pose_detection_confidence: The minimum confidence score for the pose
      detection to be considered successful.
    min_pose_presence_confidence: The minimum confidence score of pose presence
      score in the pose landmark detection.
    min_tracking_confidence: The minimum confidence score for the pose tracking
      to be considered successful.
    output_segmentation_masks: whether to output segmentation masks.
    result_callback: The user-defined result callback for processing live stream
      data. The result callback should only be specified when the running mode
      is set to the live stream mode.
  """

  base_options: _BaseOptions
  running_mode: _RunningMode = _RunningMode.IMAGE
  num_poses: int = 1
  min_pose_detection_confidence: float = 0.5
  min_pose_presence_confidence: float = 0.5
  min_tracking_confidence: float = 0.5
  output_segmentation_masks: bool = False
  result_callback: Optional[
      Callable[[PoseLandmarkerResult, image_lib.Image, int], None]
  ] = None
  _result_callback_c: Optional[
      Callable[
          [
              PoseLandmarkerResultC,
              ctypes.c_void_p,
              int,
              str,
          ],
          None,
      ]
  ] = None

  @doc_controls.do_not_generate_docs
  def to_ctypes(self) -> PoseLandmarkerOptionsC:
    """Generates an PoseLandmarkerOptionsC ctypes struct."""
    options_c = PoseLandmarkerOptionsC()
    options_c.base_options = self.base_options.to_ctypes()
    options_c.running_mode = self.running_mode.ctype
    options_c.num_poses = self.num_poses
    options_c.min_pose_detection_confidence = self.min_pose_detection_confidence
    options_c.min_pose_presence_confidence = self.min_pose_presence_confidence
    options_c.min_tracking_confidence = self.min_tracking_confidence
    options_c.output_segmentation_masks = self.output_segmentation_masks

    if self._result_callback_c is None:
      lib = mediapipe_c_bindings_lib.load_shared_library()

      # The C callback function that will be called by the C code.
      @ctypes.CFUNCTYPE(
          None,
          ctypes.POINTER(PoseLandmarkerResultC),
          ctypes.c_void_p,
          ctypes.c_int64,
          ctypes.c_char_p,
      )
      def c_callback(result, image, timestamp_ms, error_msg):
        if error_msg:
          logging.error('Pose detector error: %s', error_msg)
          return
        if self.result_callback is not None:
          py_result = PoseLandmarkerResult.from_ctypes(result)
          py_image = image_lib.Image.create_from_ctypes(image, lib)
          self.result_callback(py_result, py_image, timestamp_ms)

      # Keep callback from getting garbage collected.
      self._result_callback_c = c_callback

    options_c.result_callback = self._result_callback_c
    return options_c


def _register_ctypes_signatures(lib: ctypes.CDLL):
  """Registers C function signatures for the given library."""
  lib.pose_landmarker_create.argtypes = [
      ctypes.POINTER(PoseLandmarkerOptionsC),
      ctypes.POINTER(ctypes.c_char_p),
  ]
  lib.pose_landmarker_create.restype = ctypes.c_void_p
  lib.pose_landmarker_detect_image.argtypes = [
      ctypes.c_void_p,
      ctypes.c_void_p,
      ctypes.POINTER(PoseLandmarkerResultC),
      ctypes.POINTER(ctypes.c_char_p),
  ]
  lib.pose_landmarker_detect_image.restype = ctypes.c_int
  lib.pose_landmarker_detect_image_with_options.argtypes = [
      ctypes.c_void_p,
      ctypes.c_void_p,
      ctypes.POINTER(image_processing_options_c_lib.ImageProcessingOptionsC),
      ctypes.POINTER(PoseLandmarkerResultC),
      ctypes.POINTER(ctypes.c_char_p),
  ]
  lib.pose_landmarker_detect_image_with_options.restype = ctypes.c_int
  lib.pose_landmarker_detect_for_video.argtypes = [
      ctypes.c_void_p,
      ctypes.c_void_p,
      ctypes.c_int64,
      ctypes.POINTER(PoseLandmarkerResultC),
      ctypes.POINTER(ctypes.c_char_p),
  ]
  lib.pose_landmarker_detect_for_video.restype = ctypes.c_int
  lib.pose_landmarker_detect_for_video_with_options.argtypes = [
      ctypes.c_void_p,
      ctypes.c_void_p,
      ctypes.POINTER(image_processing_options_c_lib.ImageProcessingOptionsC),
      ctypes.c_int64,
      ctypes.POINTER(PoseLandmarkerResultC),
      ctypes.POINTER(ctypes.c_char_p),
  ]
  lib.pose_landmarker_detect_for_video_with_options.restype = ctypes.c_int
  lib.pose_landmarker_detect_async.argtypes = [
      ctypes.c_void_p,
      ctypes.c_void_p,
      ctypes.c_int64,
      ctypes.POINTER(ctypes.c_char_p),
  ]
  lib.pose_landmarker_detect_async.restype = ctypes.c_int
  lib.pose_landmarker_detect_async_with_options.argtypes = [
      ctypes.c_void_p,
      ctypes.c_void_p,
      ctypes.POINTER(image_processing_options_c_lib.ImageProcessingOptionsC),
      ctypes.c_int64,
      ctypes.POINTER(ctypes.c_char_p),
  ]
  lib.pose_landmarker_detect_async_with_options.restype = ctypes.c_int
  lib.pose_landmarker_close_result.argtypes = [
      ctypes.POINTER(PoseLandmarkerResultC)
  ]
  lib.pose_landmarker_close_result.restype = None
  lib.pose_landmarker_close.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_char_p),
  ]


class PoseLandmarker(base_vision_task_api.BaseVisionTaskApi):
  """Class that performs pose landmarks detection on images."""

  _lib: ctypes.CDLL
  _handle: ctypes.c_void_p

  def __init__(self, lib: ctypes.CDLL, handle: ctypes.c_void_p):
    self._lib = lib
    self._handle = handle

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'PoseLandmarker':
    """Creates a `PoseLandmarker` object from a model bundle file and the default `PoseLandmarkerOptions`.

    Note that the created `PoseLandmarker` instance is in image mode, for
    detecting pose landmarks on single image inputs.

    Args:
      model_path: Path to the model.

    Returns:
      `PoseLandmarker` object that's created from the model file and the
      default `PoseLandmarkerOptions`.

    Raises:
      ValueError: If failed to create `PoseLandmarker` object from the
        provided file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = PoseLandmarkerOptions(
        base_options=base_options, running_mode=_RunningMode.IMAGE
    )
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(
      cls, options: PoseLandmarkerOptions
  ) -> 'PoseLandmarker':
    """Creates the `PoseLandmarker` object from pose landmarker options.

    Args:
      options: Options for the pose landmarker task.

    Returns:
      `PoseLandmarker` object that's created from `options`.

    Raises:
      ValueError: If failed to create `PoseLandmarker` object from
        `PoseLandmarkerOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """
    base_vision_task_api.validate_running_mode(
        options.running_mode, options.result_callback
    )

    lib = mediapipe_c_bindings_lib.load_shared_library()
    _register_ctypes_signatures(lib)

    options_c = options.to_ctypes()
    error_msg = ctypes.c_char_p()
    landmarker = lib.pose_landmarker_create(
        ctypes.byref(options_c), ctypes.byref(error_msg)
    )
    if not landmarker:
      error_string = (
          error_msg.value.decode('utf-8')
          if error_msg.value is not None
          else 'Internal Error'
      )
      raise RuntimeError(f'Failed to create PoseLandmarker: {error_string}')

    return PoseLandmarker(lib, landmarker)

  def detect(
      self,
      image: image_lib.Image,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> PoseLandmarkerResult:
    """Performs pose landmarks detection on the given image.

    Only use this method when the PoseLandmarker is created with the image
    running mode.

    Args:
      image: MediaPipe Image.
      image_processing_options: Options for image processing.

    Returns:
      The pose landmarker detection results.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If pose landmarker detection failed to run.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    result_c = PoseLandmarkerResultC()
    error_msg = ctypes.c_char_p()

    if image_processing_options is not None:
      options_c = image_processing_options.to_ctypes()
      return_code = self._lib.pose_landmarker_detect_image_with_options(
          self._handle,
          c_image,
          ctypes.byref(options_c),
          ctypes.byref(result_c),
          ctypes.byref(error_msg),
      )
    else:
      return_code = self._lib.pose_landmarker_detect_image(
          self._handle,
          c_image,
          ctypes.byref(result_c),
          ctypes.byref(error_msg),
      )

    mediapipe_c_bindings_lib.handle_return_code(
        return_code, 'Pose landmark detection failed', error_msg
    )

    result = PoseLandmarkerResult.from_ctypes(result_c)
    self._lib.pose_landmarker_close_result(ctypes.byref(result_c))
    return result

  def detect_for_video(
      self,
      image: image_lib.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> PoseLandmarkerResult:
    """Performs pose landmarks detection on the provided video frame.

    Only use this method when the PoseLandmarker is created with the video
    running mode.

    Only use this method when the PoseLandmarker is created with the video
    running mode. It's required to provide the video frame's timestamp (in
    milliseconds) along with the video frame. The input timestamps should be
    monotonically increasing for adjacent calls of this method.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input video frame in milliseconds.
      image_processing_options: Options for image processing.

    Returns:
      The pose landmarker detection results.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If pose landmarker detection failed to run.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    result_c = PoseLandmarkerResultC()
    error_msg = ctypes.c_char_p()

    if image_processing_options:
      options_c = image_processing_options.to_ctypes()
      return_code = self._lib.pose_landmarker_detect_for_video_with_options(
          self._handle,
          c_image,
          ctypes.byref(options_c),
          timestamp_ms,
          ctypes.byref(result_c),
          ctypes.byref(error_msg),
      )
    else:
      return_code = self._lib.pose_landmarker_detect_for_video(
          self._handle,
          c_image,
          timestamp_ms,
          ctypes.byref(result_c),
          ctypes.byref(error_msg),
      )

    mediapipe_c_bindings_lib.handle_return_code(
        return_code, 'Pose landmark detection failed', error_msg
    )

    result = PoseLandmarkerResult.from_ctypes(result_c)
    self._lib.pose_landmarker_close_result(ctypes.byref(result_c))
    return result

  def detect_async(
      self,
      image: image_lib.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> None:
    """Sends live image data to perform pose landmarks detection.

    The results will be available via the "result_callback" provided in the
    PoseLandmarkerOptions. Only use this method when the PoseLandmarker is
    created with the live stream running mode.

    Only use this method when the PoseLandmarker is created with the live
    stream running mode. The input timestamps should be monotonically increasing
    for adjacent calls of this method. This method will return immediately after
    the input image is accepted. The results will be available via the
    `result_callback` provided in the `PoseLandmarkerOptions`. The
    `detect_async` method is designed to process live stream data such as
    camera input. To lower the overall latency, pose landmarker may drop the
    input images if needed. In other words, it's not guaranteed to have output
    per input image.

    The `result_callback` provides:
      - The pose landmarker detection results.
      - The input image that the pose landmarker runs on.
      - The input timestamp in milliseconds.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input image in milliseconds.
      image_processing_options: Options for image processing.

    Raises:
      ValueError: If the current input timestamp is smaller than what the
      pose landmarker has already processed.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    error_msg = ctypes.c_char_p()

    if image_processing_options:
      options_c = image_processing_options.to_ctypes()
      return_code = self._lib.pose_landmarker_detect_async_with_options(
          self._handle,
          c_image,
          ctypes.byref(options_c),
          timestamp_ms,
          ctypes.byref(error_msg),
      )
    else:
      return_code = self._lib.pose_landmarker_detect_async(
          self._handle, c_image, timestamp_ms, ctypes.byref(error_msg)
      )
    mediapipe_c_bindings_lib.handle_return_code(
        return_code, 'Pose landmark detection failed', error_msg
    )

  def close(self):
    """Closes the PoseLandmarker."""
    if self._handle:
      error_msg = ctypes.c_char_p()
      return_code = self._lib.pose_landmarker_close(
          self._handle, ctypes.byref(error_msg)
      )
      mediapipe_c_bindings_lib.handle_return_code(
          return_code, 'Failed to close PoseLandmarker', error_msg
      )
    self._handle = None

  def __enter__(self):
    """Returns `self` upon entering the runtime context."""
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Shuts down the MediaPipe task instance on exit of the context manager.

    Args:
      exc_type: The exception type that caused the context manager to exit.
      exc_value: The exception value that caused the context manager to exit.
      traceback: The exception traceback that caused the context manager to
        exit.

    Raises:
      RuntimeError: If the MediaPipe PoseLandmarker task failed to close.
    """
    del exc_type, exc_value, traceback  # Unused.
    self.close()
