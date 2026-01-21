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
"""MediaPipe face detector task."""

import ctypes
import dataclasses
from typing import Callable, Optional, Tuple

from mediapipe.tasks.python.components.containers import detections as detections_module
from mediapipe.tasks.python.components.containers import detections_c as detections_c_module
from mediapipe.tasks.python.core import async_result_dispatcher
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import base_options_c as base_options_c_module
from mediapipe.tasks.python.core import mediapipe_c_bindings
from mediapipe.tasks.python.core import mediapipe_c_utils
from mediapipe.tasks.python.core import serial_dispatcher
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import image as image_module
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import image_processing_options_c as image_processing_options_c_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

FaceDetectorResult = detections_module.DetectionResult
_RunningMode = running_mode_module.VisionTaskRunningMode
_BaseOptions = base_options_module.BaseOptions
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions


_C_TYPES_RESULT_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int32,  # MpStatus
    ctypes.POINTER(detections_c_module.DetectionResultC),
    ctypes.c_void_p,  # MpImage
    ctypes.c_int64,  # timestamp_ms
)


class FaceDetectorOptionsC(ctypes.Structure):
  """C types for FaceDetectorOptions."""
  _fields_ = [
      ('base_options', base_options_c_module.BaseOptionsC),
      ('running_mode', ctypes.c_int),
      ('min_detection_confidence', ctypes.c_float),
      ('min_suppression_threshold', ctypes.c_float),
      ('result_callback', _C_TYPES_RESULT_CALLBACK),
  ]

  @classmethod
  @doc_controls.do_not_generate_docs
  def from_c_options(
      cls,
      base_options: base_options_c_module.BaseOptionsC,
      running_mode: _RunningMode,
      min_detection_confidence: float,
      min_suppression_threshold: float,
      result_callback: _C_TYPES_RESULT_CALLBACK,
  ) -> 'FaceDetectorOptionsC':
    """Creates a FaceDetectorOptionsC object from the given options."""
    return cls(
        base_options=base_options,
        running_mode=running_mode.ctype,
        min_detection_confidence=min_detection_confidence,
        min_suppression_threshold=min_suppression_threshold,
        result_callback=result_callback,
    )


_CTYPES_SIGNATURES = (
    mediapipe_c_utils.CStatusFunction(
        'MpFaceDetectorCreate',
        (
            ctypes.POINTER(FaceDetectorOptionsC),
            ctypes.POINTER(ctypes.c_void_p),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpFaceDetectorDetectImage',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(
                image_processing_options_c_module.ImageProcessingOptionsC
            ),
            ctypes.POINTER(detections_c_module.DetectionResultC),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpFaceDetectorDetectForVideo',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(
                image_processing_options_c_module.ImageProcessingOptionsC
            ),
            ctypes.c_int64,
            ctypes.POINTER(detections_c_module.DetectionResultC),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpFaceDetectorDetectAsync',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(
                image_processing_options_c_module.ImageProcessingOptionsC
            ),
            ctypes.c_int64,
        ),
    ),
    mediapipe_c_utils.CFunction(
        'MpFaceDetectorCloseResult',
        [ctypes.POINTER(detections_c_module.DetectionResultC)],
        None,
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpFaceDetectorClose',
        (ctypes.c_void_p,),
    ),
)


@dataclasses.dataclass
class FaceDetectorOptions:
  """Options for the face detector task.

  Attributes:
    base_options: Base options for the face detector task.
    running_mode: The running mode of the task. Default to the image mode. Face
      detector task has three running modes: 1) The image mode for detecting
      faces on single image inputs. 2) The video mode for detecting faces on the
      decoded frames of a video. 3) The live stream mode for detecting faces on
      a live stream of input data, such as from camera.
    min_detection_confidence: The minimum confidence score for the face
      detection to be considered successful.
    min_suppression_threshold: The minimum non-maximum-suppression threshold for
      face detection to be considered overlapped.
    result_callback: The user-defined result callback for processing live stream
      data. The result callback should only be specified when the running mode
      is set to the live stream mode.
  """

  base_options: _BaseOptions
  running_mode: _RunningMode = _RunningMode.IMAGE
  min_detection_confidence: float = 0.5
  min_suppression_threshold: float = 0.3
  result_callback: Optional[
      Callable[
          [detections_module.DetectionResult, image_module.Image, int], None
      ]
  ] = None


class FaceDetector:
  """Class that performs face detection on images."""

  _lib: serial_dispatcher.SerialDispatcher
  _handle: ctypes.c_void_p
  _dispatcher: async_result_dispatcher.AsyncResultDispatcher
  _async_callback: _C_TYPES_RESULT_CALLBACK

  def __init__(
      self,
      lib: serial_dispatcher.SerialDispatcher,
      handle: ctypes.c_void_p,
      dispatcher: async_result_dispatcher.AsyncResultDispatcher,
      async_callback: _C_TYPES_RESULT_CALLBACK,
  ):
    """Initializes the face detector.

    Args:
      lib: The dispatch library to use for the face detector.
      handle: The C pointer to the face detector.
      dispatcher: The async result handler for the face detector.
      async_callback: The c callback for the face detector.
    """
    self._lib = lib
    self._handle = handle
    self._dispatcher = dispatcher
    self._async_callback = async_callback

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'FaceDetector':
    """Creates an `FaceDetector` object from a TensorFlow Lite model and the default `FaceDetectorOptions`.

    Note that the created `FaceDetector` instance is in image mode, for
    detecting faces on single image inputs.

    Args:
      model_path: Path to the model.

    Returns:
      `FaceDetector` object that's created from the model file and the default
      `FaceDetectorOptions`.

    Raises:
      ValueError: If failed to create `FaceDetector` object from the provided
        file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = FaceDetectorOptions(
        base_options=base_options, running_mode=_RunningMode.IMAGE
    )
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(cls, options: FaceDetectorOptions) -> 'FaceDetector':
    """Creates the `FaceDetector` object from face detector options.

    Args:
      options: Options for the face detector task.

    Returns:
      `FaceDetector` object that's created from `options`.

    Raises:
      ValueError: If failed to create `FaceDetector` object from
        `FaceDetectorOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """
    running_mode_module.validate_running_mode(
        options.running_mode, options.result_callback
    )

    lib = mediapipe_c_bindings.load_shared_library(_CTYPES_SIGNATURES)

    def convert_result(
        c_result_ptr: ctypes.POINTER(detections_c_module.DetectionResultC),
        image_ptr: ctypes.c_void_p,
        timestamp_ms: int,
    ) -> Tuple[FaceDetectorResult, image_module.Image, int]:
      c_result = c_result_ptr[0]
      py_result = FaceDetectorResult.from_ctypes(c_result)
      py_image = image_module.Image.create_from_ctypes(image_ptr)
      return (py_result, py_image, timestamp_ms)

    dispatcher = async_result_dispatcher.AsyncResultDispatcher(
        converter=convert_result
    )
    c_callback = dispatcher.wrap_callback(
        options.result_callback, _C_TYPES_RESULT_CALLBACK
    )
    ctypes_options = FaceDetectorOptionsC.from_c_options(
        base_options=options.base_options.to_ctypes(),
        running_mode=options.running_mode,
        min_detection_confidence=options.min_detection_confidence,
        min_suppression_threshold=options.min_suppression_threshold,
        result_callback=c_callback,
    )

    detector_handle = ctypes.c_void_p()
    lib.MpFaceDetectorCreate(
        ctypes.byref(ctypes_options),
        ctypes.byref(detector_handle),
    )

    return FaceDetector(
        lib=lib,
        handle=detector_handle,
        dispatcher=dispatcher,
        async_callback=c_callback,
    )

  def detect(
      self,
      image: image_module.Image,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> FaceDetectorResult:
    """Performs face detection on the provided MediaPipe Image.

    Only use this method when the FaceDetector is created with the image
    running mode.

    Args:
      image: MediaPipe Image.
      image_processing_options: Options for image processing.

    Returns:
      A face detection result object that contains a list of face detections,
      each detection has a bounding box that is expressed in the unrotated input
      frame of reference coordinates system, i.e. in `[0,image_width) x [0,
      image_height)`, which are the dimensions of the underlying image data.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If face detection failed to run.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    c_result = detections_c_module.DetectionResultC()

    c_image_processing_options = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpFaceDetectorDetectImage(
        self._handle,
        c_image,
        c_image_processing_options,
        ctypes.byref(c_result),
    )

    py_result = FaceDetectorResult.from_ctypes(c_result)
    self._lib.MpFaceDetectorCloseResult(ctypes.byref(c_result))
    return py_result

  def detect_for_video(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> detections_module.DetectionResult:
    """Performs face detection on the provided video frames.

    Only use this method when the FaceDetector is created with the video
    running mode. It's required to provide the video frame's timestamp (in
    milliseconds) along with the video frame. The input timestamps should be
    monotonically increasing for adjacent calls of this method.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input video frame in milliseconds.
      image_processing_options: Options for image processing.

    Returns:
      A face detection result object that contains a list of face detections,
      each detection has a bounding box that is expressed in the unrotated input
      frame of reference coordinates system, i.e. in `[0,image_width) x [0,
      image_height)`, which are the dimensions of the underlying image data.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If face detection failed to run.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    c_result = detections_c_module.DetectionResultC()

    c_image_processing_options = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpFaceDetectorDetectForVideo(
        self._handle,
        c_image,
        c_image_processing_options,
        timestamp_ms,
        ctypes.byref(c_result),
    )

    py_result = FaceDetectorResult.from_ctypes(c_result)
    self._lib.MpFaceDetectorCloseResult(ctypes.byref(c_result))
    return py_result

  def detect_async(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> None:
    """Sends live image data (an Image with a unique timestamp) to perform face detection.

    Only use this method when the FaceDetector is created with the live stream
    running mode. The input timestamps should be monotonically increasing for
    adjacent calls of this method. This method will return immediately after the
    input image is accepted. The results will be available via the
    `result_callback` provided in the `FaceDetectorOptions`. The
    `detect_async` method is designed to process live stream data such as camera
    input. To lower the overall latency, face detector may drop the input
    images if needed. In other words, it's not guaranteed to have output per
    input image.

    The `result_callback` provides:
      - A face detection result object that contains a list of face detections,
        each detection has a bounding box that is expressed in the unrotated
        input frame of reference coordinates system,
        i.e. in `[0,image_width) x [0,image_height)`, which are the dimensions
        of the underlying image data.
      - The input image that the face detector runs on.
      - The input timestamp in milliseconds.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input image in milliseconds.
      image_processing_options: Options for image processing.

    Raises:
      ValueError: If the current input timestamp is smaller than what the face
        detector has already processed.
      RuntimeError: If face detection failed to initialize.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access

    c_image_processing_options = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpFaceDetectorDetectAsync(
        self._handle,
        c_image,
        c_image_processing_options,
        timestamp_ms,
    )

  def close(self):
    """Shuts down the MediaPipe task instance."""
    if not self._handle:
      return
    self._lib.MpFaceDetectorClose(self._handle)
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
      RuntimeError: If the MediaPipe FaceDetector task failed to close.
    """
    del exc_type, exc_value, traceback  # Unused.
    self.close()

  def __del__(self):
    self.close()
