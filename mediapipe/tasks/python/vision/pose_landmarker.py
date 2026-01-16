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
import enum
from typing import Callable, Optional, Tuple

from mediapipe.tasks.python.components.containers import landmark as landmark_lib
from mediapipe.tasks.python.components.containers import landmark_c as landmark_c_lib
from mediapipe.tasks.python.core import async_result_dispatcher
from mediapipe.tasks.python.core import base_options as base_options_lib
from mediapipe.tasks.python.core import base_options_c as base_options_c_lib
from mediapipe.tasks.python.core import mediapipe_c_bindings as mediapipe_c_bindings_lib
from mediapipe.tasks.python.core import mediapipe_c_utils
from mediapipe.tasks.python.core import serial_dispatcher
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import image as image_lib
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_lib
from mediapipe.tasks.python.vision.core import image_processing_options_c as image_processing_options_c_lib
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_lib

_BaseOptions = base_options_lib.BaseOptions
_RunningMode = running_mode_lib.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_lib.ImageProcessingOptions
_AsyncResultDispatcher = async_result_dispatcher.AsyncResultDispatcher
_LiveStreamPacket = async_result_dispatcher.LiveStreamPacket


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

_C_TYPES_RESULT_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int32,  # MpStatus
    ctypes.POINTER(PoseLandmarkerResultC),
    ctypes.c_void_p,  # MpImage
    ctypes.c_int64,  # timestamp_ms
)


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
      segmentation_masks.append(image_lib.Image.create_from_ctypes(image_c))

    segmentation_masks = (
        None if c_struct.segmentation_masks_count == 0 else segmentation_masks
    )
    return cls(pose_landmarks, pose_world_landmarks, segmentation_masks)


class PoseLandmark(enum.IntEnum):
  """The 33 pose landmarks."""

  NOSE = 0
  LEFT_EYE_INNER = 1
  LEFT_EYE = 2
  LEFT_EYE_OUTER = 3
  RIGHT_EYE_INNER = 4
  RIGHT_EYE = 5
  RIGHT_EYE_OUTER = 6
  LEFT_EAR = 7
  RIGHT_EAR = 8
  MOUTH_LEFT = 9
  MOUTH_RIGHT = 10
  LEFT_SHOULDER = 11
  RIGHT_SHOULDER = 12
  LEFT_ELBOW = 13
  RIGHT_ELBOW = 14
  LEFT_WRIST = 15
  RIGHT_WRIST = 16
  LEFT_PINKY = 17
  RIGHT_PINKY = 18
  LEFT_INDEX = 19
  RIGHT_INDEX = 20
  LEFT_THUMB = 21
  RIGHT_THUMB = 22
  LEFT_HIP = 23
  RIGHT_HIP = 24
  LEFT_KNEE = 25
  RIGHT_KNEE = 26
  LEFT_ANKLE = 27
  RIGHT_ANKLE = 28
  LEFT_HEEL = 29
  RIGHT_HEEL = 30
  LEFT_FOOT_INDEX = 31
  RIGHT_FOOT_INDEX = 32


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
          _C_TYPES_RESULT_CALLBACK,
      ),
  ]

  @classmethod
  @doc_controls.do_not_generate_docs
  def from_c_options(
      cls,
      base_options: base_options_c_lib.BaseOptionsC,
      running_mode: _RunningMode,
      num_poses: int,
      min_pose_detection_confidence: float,
      min_pose_presence_confidence: float,
      min_tracking_confidence: float,
      output_segmentation_masks: bool,
      result_callback: _C_TYPES_RESULT_CALLBACK,
  ) -> 'PoseLandmarkerOptionsC':
    """Creates a PoseLandmarkerOptionsC object from the given options."""
    return cls(
        base_options=base_options,
        running_mode=running_mode.ctype,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=output_segmentation_masks,
        result_callback=result_callback,
    )


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


_CTYPES_SIGNATURES = (
    mediapipe_c_utils.CStatusFunction(
        'MpPoseLandmarkerCreate',
        (
            ctypes.POINTER(PoseLandmarkerOptionsC),
            ctypes.POINTER(ctypes.c_void_p),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpPoseLandmarkerDetectImage',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(
                image_processing_options_c_lib.ImageProcessingOptionsC
            ),
            ctypes.POINTER(PoseLandmarkerResultC),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpPoseLandmarkerDetectForVideo',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(
                image_processing_options_c_lib.ImageProcessingOptionsC
            ),
            ctypes.c_int64,
            ctypes.POINTER(PoseLandmarkerResultC),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpPoseLandmarkerDetectAsync',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(
                image_processing_options_c_lib.ImageProcessingOptionsC
            ),
            ctypes.c_int64,
        ),
    ),
    mediapipe_c_utils.CFunction(
        'MpPoseLandmarkerCloseResult',
        [ctypes.POINTER(PoseLandmarkerResultC)],
        None,
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpPoseLandmarkerClose',
        (ctypes.c_void_p,),
    ),
)


class PoseLandmarker:
  """Class that performs pose landmarks detection on images."""

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
    """Initializes the pose landmarker.

    Args:
      lib: The dispatch library to use for the pose landmarker.
      handle: The C pointer to the pose landmarker.
      dispatcher: The async result handler for the pose landmarker.
      async_callback: The c callback for the pose landmarker.
    """
    self._lib = lib
    self._handle = handle
    self._dispatcher = dispatcher
    self._async_callback = async_callback

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
    running_mode_lib.validate_running_mode(
        options.running_mode, options.result_callback
    )

    lib = mediapipe_c_bindings_lib.load_shared_library(_CTYPES_SIGNATURES)

    def convert_result(
        c_result_ptr: ctypes.POINTER(PoseLandmarkerResultC),
        image_ptr: ctypes.c_void_p,
        timestamp_ms: int,
    ) -> Tuple[PoseLandmarkerResult, image_lib.Image, int]:
      c_result = c_result_ptr[0]
      py_result = PoseLandmarkerResult.from_ctypes(c_result)
      py_image = image_lib.Image.create_from_ctypes(image_ptr)
      return (py_result, py_image, timestamp_ms)

    dispatcher = _AsyncResultDispatcher(converter=convert_result)

    c_callback = dispatcher.wrap_callback(
        options.result_callback, _C_TYPES_RESULT_CALLBACK
    )
    ctypes_options = PoseLandmarkerOptionsC.from_c_options(
        base_options=options.base_options.to_ctypes(),
        running_mode=options.running_mode,
        num_poses=options.num_poses,
        min_pose_detection_confidence=options.min_pose_detection_confidence,
        min_pose_presence_confidence=options.min_pose_presence_confidence,
        min_tracking_confidence=options.min_tracking_confidence,
        output_segmentation_masks=options.output_segmentation_masks,
        result_callback=c_callback,
    )
    landmarker = ctypes.c_void_p()
    lib.MpPoseLandmarkerCreate(
        ctypes.byref(ctypes_options), ctypes.byref(landmarker)
    )

    return PoseLandmarker(
        lib, landmarker, dispatcher=dispatcher, async_callback=c_callback
    )

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

    c_image_processing_options = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpPoseLandmarkerDetectImage(
        self._handle,
        c_image,
        c_image_processing_options,
        ctypes.byref(result_c),
    )

    result = PoseLandmarkerResult.from_ctypes(result_c)
    self._lib.MpPoseLandmarkerCloseResult(ctypes.byref(result_c))
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

    c_image_processing_options = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpPoseLandmarkerDetectForVideo(
        self._handle,
        c_image,
        c_image_processing_options,
        timestamp_ms,
        ctypes.byref(result_c),
    )

    result = PoseLandmarkerResult.from_ctypes(result_c)
    self._lib.MpPoseLandmarkerCloseResult(ctypes.byref(result_c))
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

    c_image_processing_options = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpPoseLandmarkerDetectAsync(
        self._handle,
        c_image,
        c_image_processing_options,
        timestamp_ms,
    )

  def close(self):
    """Closes the PoseLandmarker."""
    if not self._handle:
      return
    self._lib.MpPoseLandmarkerClose(self._handle)
    self._handle = None
    self._dispatcher.close()
    self._lib.close()

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

  def __del__(self):
    self.close()
