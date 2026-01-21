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
"""MediaPipe hand landmarker task."""

import ctypes
import dataclasses
import enum
from typing import Callable, List, Optional, Tuple

from mediapipe.tasks.python.components.containers import category
from mediapipe.tasks.python.components.containers import category_c
from mediapipe.tasks.python.components.containers import landmark
from mediapipe.tasks.python.components.containers import landmark_c
from mediapipe.tasks.python.core import async_result_dispatcher
from mediapipe.tasks.python.core import base_options as base_options_lib
from mediapipe.tasks.python.core import base_options_c
from mediapipe.tasks.python.core import mediapipe_c_bindings
from mediapipe.tasks.python.core import mediapipe_c_utils
from mediapipe.tasks.python.core import serial_dispatcher
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import image as image_lib
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_lib
from mediapipe.tasks.python.vision.core import image_processing_options_c
from mediapipe.tasks.python.vision.core import vision_task_running_mode

_BaseOptions = base_options_lib.BaseOptions
_RunningMode = vision_task_running_mode.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_lib.ImageProcessingOptions
_CFunction = mediapipe_c_utils.CFunction
_AsyncResultDispatcher = async_result_dispatcher.AsyncResultDispatcher


class HandLandmarkerResultC(ctypes.Structure):
  """The hand landmarker result from HandLandmarker in ctypes."""

  _fields_ = [
      ('handedness', ctypes.POINTER(category_c.CategoriesC)),
      ('handedness_count', ctypes.c_uint32),
      (
          'hand_landmarks',
          ctypes.POINTER(landmark_c.NormalizedLandmarksC),
      ),
      ('hand_landmarks_count', ctypes.c_uint32),
      ('hand_world_landmarks', ctypes.POINTER(landmark_c.LandmarksC)),
      ('hand_world_landmarks_count', ctypes.c_uint32),
  ]

_C_TYPES_RESULT_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int32,  # MpStatus
    ctypes.POINTER(HandLandmarkerResultC),
    ctypes.c_void_p,  # MpImage
    ctypes.c_int64,  # timestamp_ms
)


class HandLandmarkerOptionsC(ctypes.Structure):
  """The hand landmarker options used in the C API."""

  _fields_ = [
      ('base_options', base_options_c.BaseOptionsC),
      ('running_mode', ctypes.c_int),
      ('num_hands', ctypes.c_int),
      ('min_hand_detection_confidence', ctypes.c_float),
      ('min_hand_presence_confidence', ctypes.c_float),
      ('min_tracking_confidence', ctypes.c_float),
      ('result_callback', _C_TYPES_RESULT_CALLBACK),
  ]

  @classmethod
  @doc_controls.do_not_generate_docs
  def from_c_options(
      cls,
      base_options: base_options_c.BaseOptionsC,
      running_mode: _RunningMode,
      num_hands: int,
      min_hand_detection_confidence: float,
      min_hand_presence_confidence: float,
      min_tracking_confidence: float,
      result_callback: _C_TYPES_RESULT_CALLBACK,
  ) -> 'HandLandmarkerOptionsC':
    """Creates a HandLandmarkerOptionsC object from the given options."""
    return cls(
        base_options=base_options,
        running_mode=running_mode.ctype,
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        result_callback=result_callback,
    )


_CTYPES_SIGNATURES = (
    mediapipe_c_utils.CStatusFunction(
        'MpHandLandmarkerCreate',
        (
            ctypes.POINTER(HandLandmarkerOptionsC),
            ctypes.POINTER(ctypes.c_void_p),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpHandLandmarkerDetectImage',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(image_processing_options_c.ImageProcessingOptionsC),
            ctypes.POINTER(HandLandmarkerResultC),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpHandLandmarkerDetectForVideo',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(image_processing_options_c.ImageProcessingOptionsC),
            ctypes.c_int64,
            ctypes.POINTER(HandLandmarkerResultC),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpHandLandmarkerDetectAsync',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(image_processing_options_c.ImageProcessingOptionsC),
            ctypes.c_int64,
        ),
    ),
    _CFunction(
        'MpHandLandmarkerCloseResult',
        [ctypes.POINTER(HandLandmarkerResultC)],
        None,
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpHandLandmarkerClose',
        (ctypes.c_void_p,),
    ),
)


class HandLandmark(enum.IntEnum):
  """The 21 hand landmarks."""

  WRIST = 0
  THUMB_CMC = 1
  THUMB_MCP = 2
  THUMB_IP = 3
  THUMB_TIP = 4
  INDEX_FINGER_MCP = 5
  INDEX_FINGER_PIP = 6
  INDEX_FINGER_DIP = 7
  INDEX_FINGER_TIP = 8
  MIDDLE_FINGER_MCP = 9
  MIDDLE_FINGER_PIP = 10
  MIDDLE_FINGER_DIP = 11
  MIDDLE_FINGER_TIP = 12
  RING_FINGER_MCP = 13
  RING_FINGER_PIP = 14
  RING_FINGER_DIP = 15
  RING_FINGER_TIP = 16
  PINKY_MCP = 17
  PINKY_PIP = 18
  PINKY_DIP = 19
  PINKY_TIP = 20


class HandLandmarksConnections:
  """The connections between hand landmarks."""

  @dataclasses.dataclass
  class Connection:
    """The connection class for hand landmarks."""

    start: int
    end: int

  HAND_PALM_CONNECTIONS: List[Connection] = [
      Connection(0, 1),
      Connection(1, 5),
      Connection(9, 13),
      Connection(13, 17),
      Connection(5, 9),
      Connection(0, 17),
  ]

  HAND_THUMB_CONNECTIONS: List[Connection] = [
      Connection(1, 2),
      Connection(2, 3),
      Connection(3, 4),
  ]

  HAND_INDEX_FINGER_CONNECTIONS: List[Connection] = [
      Connection(5, 6),
      Connection(6, 7),
      Connection(7, 8),
  ]

  HAND_MIDDLE_FINGER_CONNECTIONS: List[Connection] = [
      Connection(9, 10),
      Connection(10, 11),
      Connection(11, 12),
  ]

  HAND_RING_FINGER_CONNECTIONS: List[Connection] = [
      Connection(13, 14),
      Connection(14, 15),
      Connection(15, 16),
  ]

  HAND_PINKY_FINGER_CONNECTIONS: List[Connection] = [
      Connection(17, 18),
      Connection(18, 19),
      Connection(19, 20),
  ]

  HAND_CONNECTIONS: List[Connection] = (
      HAND_PALM_CONNECTIONS +
      HAND_THUMB_CONNECTIONS +
      HAND_INDEX_FINGER_CONNECTIONS +
      HAND_MIDDLE_FINGER_CONNECTIONS +
      HAND_RING_FINGER_CONNECTIONS +
      HAND_PINKY_FINGER_CONNECTIONS
  )


@dataclasses.dataclass
class HandLandmarkerResult:
  """The hand landmarks result from HandLandmarker, where each vector element represents a single hand detected in the image.

  Attributes:
    handedness: Classification of handedness.
    hand_landmarks: Detected hand landmarks in normalized image coordinates.
    hand_world_landmarks: Detected hand landmarks in world coordinates.
  """

  handedness: List[List[category.Category]]
  hand_landmarks: List[List[landmark.NormalizedLandmark]]
  hand_world_landmarks: List[List[landmark.Landmark]]

  @classmethod
  @doc_controls.do_not_generate_docs
  def from_ctypes(cls, c_obj: HandLandmarkerResultC) -> 'HandLandmarkerResult':
    """Creates a `HandLandmarkerResult` from a `HandLandmarkerResultC`."""
    handedness = [
        category.create_list_of_categories_from_ctypes(c_obj.handedness[i])
        for i in range(c_obj.handedness_count)
    ]

    hand_landmarks = []
    for i in range(c_obj.hand_landmarks_count):
      landmarks_c = c_obj.hand_landmarks[i]
      hand_landmarks.append([
          landmark.NormalizedLandmark.from_ctypes(landmarks_c.landmarks[j])
          for j in range(landmarks_c.landmarks_count)
      ])

    hand_world_landmarks = []
    for i in range(c_obj.hand_world_landmarks_count):
      landmarks_c = c_obj.hand_world_landmarks[i]
      hand_world_landmarks.append([
          landmark.Landmark.from_ctypes(landmarks_c.landmarks[j])
          for j in range(landmarks_c.landmarks_count)
      ])

    return HandLandmarkerResult(
        handedness=handedness,
        hand_landmarks=hand_landmarks,
        hand_world_landmarks=hand_world_landmarks,
    )


@dataclasses.dataclass
class HandLandmarkerOptions:
  """Options for the hand landmarker task.

  Attributes:
    base_options: Base options for the hand landmarker task.
    running_mode: The running mode of the task. Default to the image mode.
      HandLandmarker has three running modes: 1) The image mode for detecting
      hand landmarks on single image inputs. 2) The video mode for detecting
      hand landmarks on the decoded frames of a video. 3) The live stream mode
      for detecting hand landmarks on the live stream of input data, such as
      from camera. In this mode, the "result_callback" below must be specified
      to receive the detection results asynchronously.
    num_hands: The maximum number of hands can be detected by the hand
      landmarker.
    min_hand_detection_confidence: The minimum confidence score for the hand
      detection to be considered successful.
    min_hand_presence_confidence: The minimum confidence score of hand presence
      score in the hand landmark detection.
    min_tracking_confidence: The minimum confidence score for the hand tracking
      to be considered successful.
    result_callback: The user-defined result callback for processing live stream
      data. The result callback should only be specified when the running mode
      is set to the live stream mode.
  """

  base_options: _BaseOptions
  running_mode: _RunningMode = _RunningMode.IMAGE
  num_hands: int = 1
  min_hand_detection_confidence: float = 0.5
  min_hand_presence_confidence: float = 0.5
  min_tracking_confidence: float = 0.5
  result_callback: Optional[
      Callable[[HandLandmarkerResult, image_lib.Image, int], None]
  ] = None


class HandLandmarker:
  """Class that performs hand landmarks detection on images."""

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
    """Initializes the hand landmarker.

    Args:
      lib: The dispatch library to use for the hand landmarker.
      handle: The C pointer to the hand landmarker.
      dispatcher: The async result handler for the hand landmarker.
      async_callback: The c callback for the hand landmarker.
    """
    self._lib = lib
    self._handle = handle
    self._dispatcher = dispatcher
    self._async_callback = async_callback

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'HandLandmarker':
    """Creates an `HandLandmarker` object from a TensorFlow Lite model and the default `HandLandmarkerOptions`.

    Note that the created `HandLandmarker` instance is in image mode, for
    detecting hand landmarks on single image inputs.

    Args:
      model_path: Path to the model.

    Returns:
      `HandLandmarker` object that's created from the model file and the
      default `HandLandmarkerOptions`.

    Raises:
      ValueError: If failed to create `HandLandmarker` object from the
        provided file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    options = HandLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=model_path),
        running_mode=_RunningMode.IMAGE,
    )
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(
      cls, options: HandLandmarkerOptions
  ) -> 'HandLandmarker':
    """Creates the `HandLandmarker` object from hand landmarker options.

    Args:
      options: Options for the hand landmarker task.

    Returns:
      `HandLandmarker` object that's created from `options`.

    Raises:
      ValueError: If failed to create `HandLandmarker` object from
        `HandLandmarkerOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """
    vision_task_running_mode.validate_running_mode(
        options.running_mode, options.result_callback
    )

    lib = mediapipe_c_bindings.load_shared_library(_CTYPES_SIGNATURES)

    def convert_result(
        c_result_ptr: ctypes.POINTER(HandLandmarkerResultC),
        image_ptr: ctypes.c_void_p,
        timestamp_ms: int,
    ) -> Tuple[HandLandmarkerResult, image_lib.Image, int]:
      c_result = c_result_ptr[0]
      py_result = HandLandmarkerResult.from_ctypes(c_result)
      py_image = image_lib.Image.create_from_ctypes(image_ptr)
      return (py_result, py_image, timestamp_ms)

    dispatcher = _AsyncResultDispatcher(converter=convert_result)
    c_callback = dispatcher.wrap_callback(
        options.result_callback, _C_TYPES_RESULT_CALLBACK
    )
    ctypes_options = HandLandmarkerOptionsC.from_c_options(
        base_options=options.base_options.to_ctypes(),
        running_mode=options.running_mode,
        num_hands=options.num_hands,
        min_hand_detection_confidence=options.min_hand_detection_confidence,
        min_hand_presence_confidence=options.min_hand_presence_confidence,
        min_tracking_confidence=options.min_tracking_confidence,
        result_callback=c_callback,
    )

    landmarker_handle = ctypes.c_void_p()
    lib.MpHandLandmarkerCreate(
        ctypes.byref(ctypes_options),
        ctypes.byref(landmarker_handle),
    )
    return HandLandmarker(
        lib=lib,
        handle=landmarker_handle,
        dispatcher=dispatcher,
        async_callback=c_callback,
    )

  def detect(
      self,
      image: image_lib.Image,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> HandLandmarkerResult:
    """Performs hand landmarks detection on the given image.

    Only use this method when the HandLandmarker is created with the image
    running mode.

    The image can be of any size with format RGB or RGBA.
    TODO: Describes how the input image will be preprocessed after the yuv
    support is implemented.

    Args:
      image: MediaPipe Image.
      image_processing_options: Options for image processing.

    Returns:
      The hand landmarks detection results.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If hand landmarker detection failed to run.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    c_result = HandLandmarkerResultC()

    c_image_processing_options = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpHandLandmarkerDetectImage(
        self._handle,
        c_image,
        c_image_processing_options,
        ctypes.byref(c_result),
    )

    py_result = HandLandmarkerResult.from_ctypes(c_result)
    self._lib.MpHandLandmarkerCloseResult(ctypes.byref(c_result))
    return py_result

  def detect_for_video(
      self,
      image: image_lib.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> HandLandmarkerResult:
    """Performs hand landmarks detection on the provided video frame.

    Only use this method when the HandLandmarker is created with the video
    running mode.

    Only use this method when the HandLandmarker is created with the video
    running mode. It's required to provide the video frame's timestamp (in
    milliseconds) along with the video frame. The input timestamps should be
    monotonically increasing for adjacent calls of this method.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input video frame in milliseconds.
      image_processing_options: Options for image processing.

    Returns:
      The hand landmarks detection results.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If hand landmarker detection failed to run.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    c_result = HandLandmarkerResultC()

    c_image_processing_options = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpHandLandmarkerDetectForVideo(
        self._handle,
        c_image,
        c_image_processing_options,
        timestamp_ms,
        ctypes.byref(c_result),
    )

    py_result = HandLandmarkerResult.from_ctypes(c_result)
    self._lib.MpHandLandmarkerCloseResult(ctypes.byref(c_result))
    return py_result

  def detect_async(
      self,
      image: image_lib.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> None:
    """Sends live image data to perform hand landmarks detection.

    The results will be available via the "result_callback" provided in the
    HandLandmarkerOptions. Only use this method when the HandLandmarker is
    created with the live stream running mode.

    Only use this method when the HandLandmarker is created with the live
    stream running mode. The input timestamps should be monotonically increasing
    for adjacent calls of this method. This method will return immediately after
    the input image is accepted. The results will be available via the
    `result_callback` provided in the `HandLandmarkerOptions`. The
    `detect_async` method is designed to process live stream data such as
    camera input. To lower the overall latency, hand landmarker may drop the
    input images if needed. In other words, it's not guaranteed to have output
    per input image.

    The `result_callback` provides:
      - The hand landmarks detection results.
      - The input image that the hand landmarker runs on.
      - The input timestamp in milliseconds.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input image in milliseconds.
      image_processing_options: Options for image processing.

    Raises:
      ValueError: If the current input timestamp is smaller than what the
        hand landmarker has already processed.
      RuntimeError: If hand landmarker detection failed to initialize.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access

    c_image_processing_options = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpHandLandmarkerDetectAsync(
        self._handle,
        c_image,
        c_image_processing_options,
        timestamp_ms,
    )

  def close(self):
    """Shuts down the MediaPipe task instance."""
    if self._handle:
      self._lib.MpHandLandmarkerClose(self._handle)
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
      RuntimeError: If the MediaPipe HandLandmarker task failed to close.
    """
    del exc_type, exc_value, traceback  # Unused.
    self.close()

  def __del__(self):
    self.close()
