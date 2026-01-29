# Copyright 2023 The MediaPipe Authors.
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
"""MediaPipe holistic landmarker task."""

import ctypes
import dataclasses
from typing import Callable, List, Optional, Tuple

from mediapipe.tasks.python.components.containers import category as category_lib
from mediapipe.tasks.python.components.containers import category_c as category_c_lib
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


class HolisticLandmarkerResultC(ctypes.Structure):
  """The ctypes struct for HolisticLandmarkerResult."""

  _fields_ = [
      ('face_landmarks', landmark_c_lib.NormalizedLandmarksC),
      ('pose_landmarks', landmark_c_lib.NormalizedLandmarksC),
      ('pose_world_landmarks', landmark_c_lib.LandmarksC),
      ('left_hand_landmarks', landmark_c_lib.NormalizedLandmarksC),
      ('right_hand_landmarks', landmark_c_lib.NormalizedLandmarksC),
      ('left_hand_world_landmarks', landmark_c_lib.LandmarksC),
      ('right_hand_world_landmarks', landmark_c_lib.LandmarksC),
      ('face_blendshapes', category_c_lib.CategoriesC),
      ('pose_segmentation_mask', ctypes.c_void_p),
  ]


_C_TYPES_RESULT_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int32,  # MpStatus
    ctypes.POINTER(HolisticLandmarkerResultC),
    ctypes.c_void_p,  # MpImage
    ctypes.c_int64,  # timestamp_ms
)


@dataclasses.dataclass
class HolisticLandmarkerResult:
  """The holistic landmarks result from HolisticLandmarker, where each vector element represents a single holistic detected in the image.

  Attributes:
    face_landmarks: Detected face landmarks in normalized image coordinates.
    pose_landmarks: Detected pose landmarks in normalized image coordinates.
    pose_world_landmarks: Detected pose world landmarks in image coordinates.
    left_hand_landmarks: Detected left hand landmarks in normalized image
      coordinates.
    left_hand_world_landmarks: Detected left hand landmarks in image
      coordinates.
    right_hand_landmarks: Detected right hand landmarks in normalized image
      coordinates.
    right_hand_world_landmarks: Detected right hand landmarks in image
      coordinates.
    face_blendshapes: Optional face blendshapes.
    segmentation_mask: Optional segmentation mask for pose.
  """

  face_landmarks: List[landmark_lib.NormalizedLandmark]
  pose_landmarks: List[landmark_lib.NormalizedLandmark]
  pose_world_landmarks: List[landmark_lib.Landmark]
  left_hand_landmarks: List[landmark_lib.NormalizedLandmark]
  left_hand_world_landmarks: List[landmark_lib.Landmark]
  right_hand_landmarks: List[landmark_lib.NormalizedLandmark]
  right_hand_world_landmarks: List[landmark_lib.Landmark]
  face_blendshapes: Optional[List[category_lib.Category]] = None
  segmentation_mask: Optional[image_lib.Image] = None

  @classmethod
  @doc_controls.do_not_generate_docs
  def from_ctypes(
      cls, c_struct: HolisticLandmarkerResultC
  ) -> 'HolisticLandmarkerResult':
    """Creates a `HolisticLandmarkerResult` object from the given ctypes struct."""
    face_landmarks = [
        landmark_lib.NormalizedLandmark.from_ctypes(
            c_struct.face_landmarks.landmarks[i]
        )
        for i in range(c_struct.face_landmarks.landmarks_count)
    ]
    pose_landmarks = [
        landmark_lib.NormalizedLandmark.from_ctypes(
            c_struct.pose_landmarks.landmarks[i]
        )
        for i in range(c_struct.pose_landmarks.landmarks_count)
    ]
    pose_world_landmarks = [
        landmark_lib.Landmark.from_ctypes(
            c_struct.pose_world_landmarks.landmarks[i]
        )
        for i in range(c_struct.pose_world_landmarks.landmarks_count)
    ]
    left_hand_landmarks = [
        landmark_lib.NormalizedLandmark.from_ctypes(
            c_struct.left_hand_landmarks.landmarks[i]
        )
        for i in range(c_struct.left_hand_landmarks.landmarks_count)
    ]
    left_hand_world_landmarks = [
        landmark_lib.Landmark.from_ctypes(
            c_struct.left_hand_world_landmarks.landmarks[i]
        )
        for i in range(c_struct.left_hand_world_landmarks.landmarks_count)
    ]
    right_hand_landmarks = [
        landmark_lib.NormalizedLandmark.from_ctypes(
            c_struct.right_hand_landmarks.landmarks[i]
        )
        for i in range(c_struct.right_hand_landmarks.landmarks_count)
    ]
    right_hand_world_landmarks = [
        landmark_lib.Landmark.from_ctypes(
            c_struct.right_hand_world_landmarks.landmarks[i]
        )
        for i in range(c_struct.right_hand_world_landmarks.landmarks_count)
    ]
    face_blendshapes = [
        category_lib.Category.from_ctypes(
            c_struct.face_blendshapes.categories[i]
        )
        for i in range(c_struct.face_blendshapes.categories_count)
    ]
    return cls(
        face_landmarks=face_landmarks,
        pose_landmarks=pose_landmarks,
        pose_world_landmarks=pose_world_landmarks,
        left_hand_landmarks=left_hand_landmarks,
        right_hand_landmarks=right_hand_landmarks,
        left_hand_world_landmarks=left_hand_world_landmarks,
        right_hand_world_landmarks=right_hand_world_landmarks,
        face_blendshapes=(
            face_blendshapes
            if c_struct.face_blendshapes.categories_count > 0
            else None
        ),
        segmentation_mask=(
            image_lib.Image.create_from_ctypes(c_struct.pose_segmentation_mask)
            if c_struct.pose_segmentation_mask
            else None
        ),
    )


class HolisticLandmarkerOptionsC(ctypes.Structure):
  """The ctypes struct for HolisticLandmarkerOptions."""

  _fields_ = [
      ('base_options', base_options_c_lib.BaseOptionsC),
      ('running_mode', ctypes.c_int),
      ('min_face_detection_confidence', ctypes.c_float),
      ('min_face_suppression_threshold', ctypes.c_float),
      ('min_face_landmarks_confidence', ctypes.c_float),
      ('min_pose_detection_confidence', ctypes.c_float),
      ('min_pose_suppression_threshold', ctypes.c_float),
      ('min_pose_landmarks_confidence', ctypes.c_float),
      ('min_hand_landmarks_confidence', ctypes.c_float),
      ('output_face_blendshapes', ctypes.c_bool),
      ('output_segmentation_masks', ctypes.c_bool),
      ('result_callback', _C_TYPES_RESULT_CALLBACK),
  ]

  @classmethod
  @doc_controls.do_not_generate_docs
  def from_c_options(
      cls,
      base_options: base_options_c_lib.BaseOptionsC,
      running_mode: _RunningMode,
      min_face_detection_confidence: float,
      min_face_suppression_threshold: float,
      min_face_landmarks_confidence: float,
      min_pose_detection_confidence: float,
      min_pose_suppression_threshold: float,
      min_pose_landmarks_confidence: float,
      min_hand_landmarks_confidence: float,
      output_face_blendshapes: bool,
      output_segmentation_masks: bool,
      result_callback: _C_TYPES_RESULT_CALLBACK,
  ) -> 'HolisticLandmarkerOptionsC':
    """Creates a HolisticLandmarkerOptionsC object from the given options."""
    return cls(
        base_options=base_options,
        running_mode=running_mode.ctype,
        min_face_detection_confidence=min_face_detection_confidence,
        min_face_suppression_threshold=min_face_suppression_threshold,
        min_face_landmarks_confidence=min_face_landmarks_confidence,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_suppression_threshold=min_pose_suppression_threshold,
        min_pose_landmarks_confidence=min_pose_landmarks_confidence,
        min_hand_landmarks_confidence=min_hand_landmarks_confidence,
        output_face_blendshapes=output_face_blendshapes,
        output_segmentation_masks=output_segmentation_masks,
        result_callback=result_callback,
    )


@dataclasses.dataclass
class HolisticLandmarkerOptions:
  """Options for the holistic landmarker task.

  Attributes:
    base_options: Base options for the holistic landmarker task.
    running_mode: The running mode of the task. Default to the image mode.
      HolisticLandmarker has three running modes: 1) The image mode for
      detecting holistic landmarks on single image inputs. 2) The video mode for
      detecting holistic landmarks on the decoded frames of a video. 3) The live
      stream mode for detecting holistic landmarks on the live stream of input
      data, such as from camera. In this mode, the "result_callback" below must
      be specified to receive the detection results asynchronously.
    min_face_detection_confidence: The minimum confidence score for the face
      detection to be considered successful.
    min_face_suppression_threshold: The minimum non-maximum-suppression
      threshold for face detection to be considered overlapped.
    min_face_landmarks_confidence: The minimum confidence score for the face
      landmark detection to be considered successful.
    min_pose_detection_confidence: The minimum confidence score for the pose
      detection to be considered successful.
    min_pose_suppression_threshold: The minimum non-maximum-suppression
      threshold for pose detection to be considered overlapped.
    min_pose_landmarks_confidence: The minimum confidence score for the pose
      landmark detection to be considered successful.
    min_hand_landmarks_confidence: The minimum confidence score for the hand
      landmark detection to be considered successful.
    output_face_blendshapes: Whether HolisticLandmarker outputs face blendshapes
      classification. Face blendshapes are used for rendering the 3D face model.
    output_segmentation_mask: whether to output segmentation masks.
    result_callback: The user-defined result callback for processing live stream
      data. The result callback should only be specified when the running mode
      is set to the live stream mode.
  """

  base_options: _BaseOptions
  running_mode: _RunningMode = _RunningMode.IMAGE
  min_face_detection_confidence: float = 0.5
  min_face_suppression_threshold: float = 0.5
  min_face_landmarks_confidence: float = 0.5
  min_pose_detection_confidence: float = 0.5
  min_pose_suppression_threshold: float = 0.5
  min_pose_landmarks_confidence: float = 0.5
  min_hand_landmarks_confidence: float = 0.5
  output_face_blendshapes: bool = False
  output_segmentation_mask: bool = False
  result_callback: Optional[
      Callable[[HolisticLandmarkerResult, image_lib.Image, int], None]
  ] = None


_CTYPES_SIGNATURES = (
    mediapipe_c_utils.CStatusFunction(
        'MpHolisticLandmarkerCreate',
        (
            ctypes.POINTER(HolisticLandmarkerOptionsC),
            ctypes.POINTER(ctypes.c_void_p),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpHolisticLandmarkerDetectImage',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(
                image_processing_options_c_lib.ImageProcessingOptionsC
            ),
            ctypes.POINTER(HolisticLandmarkerResultC),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpHolisticLandmarkerDetectForVideo',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(
                image_processing_options_c_lib.ImageProcessingOptionsC
            ),
            ctypes.c_int64,
            ctypes.POINTER(HolisticLandmarkerResultC),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpHolisticLandmarkerDetectAsync',
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
        'MpHolisticLandmarkerCloseResult',
        [ctypes.POINTER(HolisticLandmarkerResultC)],
        None,
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpHolisticLandmarkerClose',
        (ctypes.c_void_p,),
    ),
)


class HolisticLandmarker:
  """Class that performs holistic landmarks detection on images."""

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
    """Initializes the holistic landmarker.

    Args:
      lib: The dispatch library to use for the holistic landmarker.
      handle: The C pointer to the holistic landmarker.
      dispatcher: The async result handler for the holistic landmarker.
      async_callback: The c callback for the holistic landmarker.
    """
    self._lib = lib
    self._handle = handle
    self._dispatcher = dispatcher
    self._async_callback = async_callback

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'HolisticLandmarker':
    """Creates an `HolisticLandmarker` object from a TensorFlow Lite model and the default `HolisticLandmarkerOptions`.

    Note that the created `HolisticLandmarker` instance is in image mode, for
    detecting holistic landmarks on single image inputs.

    Args:
      model_path: Path to the model.

    Returns:
      `HolisticLandmarker` object that's created from the model file and the
      default `HolisticLandmarkerOptions`.

    Raises:
      ValueError: If failed to create `HolisticLandmarker` object from the
        provided file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = HolisticLandmarkerOptions(
        base_options=base_options, running_mode=_RunningMode.IMAGE
    )
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(
      cls, options: HolisticLandmarkerOptions
  ) -> 'HolisticLandmarker':
    """Creates the `HolisticLandmarker` object from holistic landmarker options.

    Args:
      options: Options for the holistic landmarker task.

    Returns:
      `HolisticLandmarker` object that's created from `options`.

    Raises:
      ValueError: If failed to create `HolisticLandmarker` object from
        `HolisticLandmarkerOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """
    running_mode_lib.validate_running_mode(
        options.running_mode, options.result_callback
    )
    lib = mediapipe_c_bindings_lib.load_shared_library(_CTYPES_SIGNATURES)

    def convert_result(
        c_result_ptr: ctypes.POINTER(HolisticLandmarkerResultC),
        image_ptr: ctypes.c_void_p,
        timestamp_ms: int,
    ) -> Tuple[HolisticLandmarkerResult, image_lib.Image, int]:
      c_result = c_result_ptr[0]
      py_result = HolisticLandmarkerResult.from_ctypes(c_result)
      py_image = image_lib.Image.create_from_ctypes(image_ptr)
      return py_result, py_image, timestamp_ms

    dispatcher = _AsyncResultDispatcher(converter=convert_result)
    c_callback = dispatcher.wrap_callback(
        options.result_callback, _C_TYPES_RESULT_CALLBACK
    )
    options_c = HolisticLandmarkerOptionsC.from_c_options(
        base_options=options.base_options.to_ctypes(),
        running_mode=options.running_mode,
        min_face_detection_confidence=options.min_face_detection_confidence,
        min_face_suppression_threshold=options.min_face_suppression_threshold,
        min_face_landmarks_confidence=options.min_face_landmarks_confidence,
        min_pose_detection_confidence=options.min_pose_detection_confidence,
        min_pose_suppression_threshold=options.min_pose_suppression_threshold,
        min_pose_landmarks_confidence=options.min_pose_landmarks_confidence,
        min_hand_landmarks_confidence=options.min_hand_landmarks_confidence,
        output_face_blendshapes=options.output_face_blendshapes,
        output_segmentation_masks=options.output_segmentation_mask,
        result_callback=c_callback,
    )
    landmarker = ctypes.c_void_p()
    lib.MpHolisticLandmarkerCreate(
        ctypes.byref(options_c), ctypes.byref(landmarker)
    )
    return HolisticLandmarker(
        lib, landmarker, dispatcher=dispatcher, async_callback=c_callback
    )

  def detect(
      self,
      image: image_lib.Image,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> HolisticLandmarkerResult:
    """Performs holistic landmarks detection on the given image.

    Only use this method when the HolisticLandmarker is created with the image
    running mode.

    The image can be of any size with format RGB or RGBA.

    Args:
      image: MediaPipe Image.
      image_processing_options: Options for image processing.

    Returns:
      The holistic landmarks detection results.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If holistic landmarker detection failed to run.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    result_c = HolisticLandmarkerResultC()
    c_image_processing_options = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpHolisticLandmarkerDetectImage(
        self._handle,
        c_image,
        c_image_processing_options,
        ctypes.byref(result_c),
    )
    try:
      result = HolisticLandmarkerResult.from_ctypes(result_c)
    finally:
      self._lib.MpHolisticLandmarkerCloseResult(ctypes.byref(result_c))
    return result

  def detect_for_video(
      self,
      image: image_lib.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> HolisticLandmarkerResult:
    """Performs holistic landmarks detection on the provided video frame.

    Only use this method when the HolisticLandmarker is created with the video
    running mode.

    Only use this method when the HolisticLandmarker is created with the video
    running mode. It's required to provide the video frame's timestamp (in
    milliseconds) along with the video frame. The input timestamps should be
    monotonically increasing for adjacent calls of this method.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input video frame in milliseconds.
      image_processing_options: Options for image processing.

    Returns:
      The holistic landmarks detection results.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If holistic landmarker detection failed to run.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    result_c = HolisticLandmarkerResultC()
    c_image_processing_options = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpHolisticLandmarkerDetectForVideo(
        self._handle,
        c_image,
        c_image_processing_options,
        timestamp_ms,
        ctypes.byref(result_c),
    )
    try:
      result = HolisticLandmarkerResult.from_ctypes(result_c)
    finally:
      self._lib.MpHolisticLandmarkerCloseResult(ctypes.byref(result_c))
    return result

  def detect_async(
      self,
      image: image_lib.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> None:
    """Sends live image data to perform holistic landmarks detection.

    The results will be available via the "result_callback" provided in the
    HolisticLandmarkerOptions. Only use this method when the HolisticLandmarker
    is
    created with the live stream running mode.

    Only use this method when the HolisticLandmarker is created with the live
    stream running mode. The input timestamps should be monotonically increasing
    for adjacent calls of this method. This method will return immediately after
    the input image is accepted. The results will be available via the
    `result_callback` provided in the `HolisticLandmarkerOptions`. The
    `detect_async` method is designed to process live stream data such as
    camera input. To lower the overall latency, holistic landmarker may drop the
    input images if needed. In other words, it's not guaranteed to have output
    per input image.

    The `result_callback` provides:
      - The holistic landmarks detection results.
      - The input image that the holistic landmarker runs on.
      - The input timestamp in milliseconds.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input image in milliseconds.
      image_processing_options: Options for image processing.

    Raises:
      ValueError: If the current input timestamp is smaller than what the
      holistic landmarker has already processed.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    c_image_processing_options = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpHolisticLandmarkerDetectAsync(
        self._handle,
        c_image,
        c_image_processing_options,
        timestamp_ms,
    )

  def close(self):
    """Closes the HolisticLandmarker."""
    if not self._handle:
      return
    self._lib.MpHolisticLandmarkerClose(self._handle)
    self._handle = None
    self._dispatcher.close()
    self._lib.close()

  def __enter__(self):
    """Returns `self` upon entering the runtime context."""
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Shuts down the MediaPipe task instance on exit of the context manager."""
    del exc_type, exc_value, traceback  # Unused.
    self.close()

  def __del__(self):
    self.close()
