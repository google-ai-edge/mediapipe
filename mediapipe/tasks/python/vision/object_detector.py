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
"""MediaPipe object detector task."""

import ctypes
import dataclasses
import logging
from typing import Callable, List, Optional

from mediapipe.tasks.python.components.containers import detections as detections_module
from mediapipe.tasks.python.components.containers import detections_c as detections_c_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import base_options_c as base_options_c_module
from mediapipe.tasks.python.core import mediapipe_c_bindings as mediapipe_c_bindings_c_module
from mediapipe.tasks.python.core import serial_dispatcher
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import base_vision_task_api
from mediapipe.tasks.python.vision.core import image as image_module
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import image_processing_options_c as image_processing_options_c_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

ObjectDetectorResult = detections_module.DetectionResult
_BaseOptions = base_options_module.BaseOptions
_RunningMode = running_mode_module.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions
_CFunction = mediapipe_c_bindings_c_module.CFunction


class ObjectDetectorOptionsC(ctypes.Structure):
  """The object detector options used in the C API."""

  _fields_ = [
      ('base_options', base_options_c_module.BaseOptionsC),
      ('running_mode', ctypes.c_int),
      ('display_names_locale', ctypes.c_char_p),
      ('max_results', ctypes.c_int),
      ('score_threshold', ctypes.c_float),
      ('category_allowlist', ctypes.POINTER(ctypes.c_char_p)),
      ('category_allowlist_count', ctypes.c_int),
      ('category_denylist', ctypes.POINTER(ctypes.c_char_p)),
      ('category_denylist_count', ctypes.c_int),
      (
          'result_callback',
          ctypes.CFUNCTYPE(
              None,
              ctypes.POINTER(detections_c_module.DetectionResultC),
              ctypes.c_void_p,  # image
              ctypes.c_int64,  # timestamp_ms
              ctypes.c_char_p,  # error_msg
          ),
      ),
  ]


_CTYPES_SIGNATURES = (
    _CFunction(
        'object_detector_create',
        [
            ctypes.POINTER(ObjectDetectorOptionsC),
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_void_p,
    ),
    _CFunction(
        'object_detector_detect_image',
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(detections_c_module.DetectionResultC),
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_int,
    ),
    _CFunction(
        'object_detector_detect_image_with_options',
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(
                image_processing_options_c_module.ImageProcessingOptionsC
            ),
            ctypes.POINTER(detections_c_module.DetectionResultC),
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_int,
    ),
    _CFunction(
        'object_detector_detect_for_video',
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.POINTER(detections_c_module.DetectionResultC),
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_int,
    ),
    _CFunction(
        'object_detector_detect_for_video_with_options',
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(
                image_processing_options_c_module.ImageProcessingOptionsC
            ),
            ctypes.c_int64,
            ctypes.POINTER(detections_c_module.DetectionResultC),
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_int,
    ),
    _CFunction(
        'object_detector_detect_async',
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_int,
    ),
    _CFunction(
        'object_detector_detect_async_with_options',
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(
                image_processing_options_c_module.ImageProcessingOptionsC
            ),
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_int,
    ),
    _CFunction(
        'object_detector_close_result',
        [ctypes.POINTER(detections_c_module.DetectionResultC)],
        None,
    ),
    _CFunction(
        'object_detector_close',
        [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_int,
    ),
)


@dataclasses.dataclass
class ObjectDetectorOptions:
  """Options for the object detector task.

  Attributes:
    base_options: Base options for the object detector task.
    running_mode: The running mode of the task. Default to the image mode.
      Object detector task has three running modes: 1) The image mode for
      detecting objects on single image inputs. 2) The video mode for detecting
      objects on the decoded frames of a video. 3) The live stream mode for
      detecting objects on a live stream of input data, such as from camera.
    display_names_locale: The locale to use for display names specified through
      the TFLite Model Metadata.
    max_results: The maximum number of top-scored classification results to
      return.
    score_threshold: Overrides the ones provided in the model metadata. Results
      below this value are rejected.
    category_allowlist: Allowlist of category names. If non-empty, detection
      results whose category name is not in this set will be filtered out.
      Duplicate or unknown category names are ignored. Mutually exclusive with
      `category_denylist`.
    category_denylist: Denylist of category names. If non-empty, detection
      results whose category name is in this set will be filtered out. Duplicate
      or unknown category names are ignored. Mutually exclusive with
      `category_allowlist`.
    result_callback: The user-defined result callback for processing live stream
      data. The result callback should only be specified when the running mode
      is set to the live stream mode.
  """

  base_options: _BaseOptions
  running_mode: _RunningMode = _RunningMode.IMAGE
  display_names_locale: Optional[str] = None
  max_results: Optional[int] = -1
  score_threshold: Optional[float] = 0.0
  category_allowlist: Optional[List[str]] = None
  category_denylist: Optional[List[str]] = None
  result_callback: Optional[
      Callable[
          [detections_module.DetectionResult, image_module.Image, int], None
      ]
  ] = None

  _result_callback_c: Optional[
      Callable[
          [detections_c_module.DetectionResultC, ctypes.c_void_p, int, str],
          None,
      ]
  ] = None

  @doc_controls.do_not_generate_docs
  def to_ctypes(self) -> ObjectDetectorOptionsC:
    """Generates a ObjectDetectorOptionsC object."""
    if self._result_callback_c is None:
      # The C callback function that will be called by the C code.
      @ctypes.CFUNCTYPE(
          None,
          ctypes.POINTER(detections_c_module.DetectionResultC),
          ctypes.c_void_p,
          ctypes.c_int64,
          ctypes.c_char_p,
      )
      def c_callback(result, image, timestamp_ms, error_msg):
        if error_msg:
          logging.error('Object detector error: %s', error_msg)
          return

        if self.result_callback:
          py_result = detections_module.DetectionResult.from_ctypes(result)
          py_image = image_module.Image.create_from_ctypes(image)
          self.result_callback(py_result, py_image, timestamp_ms)

      # Keep callback from getting garbage collected.
      self._result_callback_c = c_callback

    category_allowlist_c = (
        mediapipe_c_bindings_c_module.convert_strings_to_ctypes_array(
            self.category_allowlist
        )
    )
    category_denylist_c = (
        mediapipe_c_bindings_c_module.convert_strings_to_ctypes_array(
            self.category_denylist
        )
    )

    category_allowlist_count_c = (
        len(self.category_allowlist) if self.category_allowlist else 0
    )
    category_denylist_count_c = (
        len(self.category_denylist) if self.category_denylist else 0
    )

    base_options_c = self.base_options.to_ctypes()
    return ObjectDetectorOptionsC(
        base_options=base_options_c,
        running_mode=self.running_mode.ctype,
        display_names_locale=self.display_names_locale,
        max_results=self.max_results,
        score_threshold=self.score_threshold,
        category_allowlist=category_allowlist_c,
        category_allowlist_count=category_allowlist_count_c,
        category_denylist=category_denylist_c,
        category_denylist_count=category_denylist_count_c,
        result_callback=self._result_callback_c,
    )


class ObjectDetector:
  """Class that performs object detection on images.

  The API expects a TFLite model with mandatory TFLite Model Metadata.

  Input tensor:
    (kTfLiteUInt8/kTfLiteFloat32)
    - image input of size `[batch x height x width x channels]`.
    - batch inference is not supported (`batch` is required to be 1).
    - only RGB inputs are supported (`channels` is required to be 3).
    - if type is kTfLiteFloat32, NormalizationOptions are required to be
      attached to the metadata for input normalization.
  Output tensors must be the 4 outputs of a `DetectionPostProcess` op, i.e:
    (kTfLiteFloat32)
    - locations tensor of size `[num_results x 4]`, the inner array
      representing bounding boxes in the form [top, left, right, bottom].
    - BoundingBoxProperties are required to be attached to the metadata
      and must specify type=BOUNDARIES and coordinate_type=RATIO.
    (kTfLiteFloat32)
    - classes tensor of size `[num_results]`, each value representing the
      integer index of a class.
    - optional (but recommended) label map(s) can be attached as
      AssociatedFile-s with type TENSOR_VALUE_LABELS, containing one label per
      line. The first such AssociatedFile (if any) is used to fill the
      `class_name` field of the results. The `display_name` field is filled
      from the AssociatedFile (if any) whose locale matches the
      `display_names_locale` field of the `ObjectDetectorOptions` used at
      creation time ("en" by default, i.e. English). If none of these are
      available, only the `index` field of the results will be filled.
    (kTfLiteFloat32)
    - scores tensor of size `[num_results]`, each value representing the score
      of the detected object.
    - optional score calibration can be attached using ScoreCalibrationOptions
      and an AssociatedFile with type TENSOR_AXIS_SCORE_CALIBRATION. See
      metadata_schema.fbs [1] for more details.
    (kTfLiteFloat32)
    - integer num_results as a tensor of size `[1]`

  An example of such model can be found at:
  https://tfhub.dev/google/lite-model/object_detection/mobile_object_localizer_v1/1/metadata/1

  [1]:
  https://github.com/google/mediapipe/blob/6cdc6443b6a7ed662744e2a2ce2d58d9c83e6d6f/mediapipe/tasks/metadata/metadata_schema.fbs#L456
  """

  _lib: serial_dispatcher.SerialDispatcher
  _handle: ctypes.c_void_p

  def __init__(
      self,
      lib: serial_dispatcher.SerialDispatcher,
      handle: ctypes.c_void_p,
  ):
    self._lib = lib
    self._handle = handle

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'ObjectDetector':
    """Creates an `ObjectDetector` object from a TensorFlow Lite model and the default `ObjectDetectorOptions`.

    Note that the created `ObjectDetector` instance is in image mode, for
    detecting objects on single image inputs.

    Args:
      model_path: Path to the model.

    Returns:
      `ObjectDetector` object that's created from the model file and the default
      `ObjectDetectorOptions`.

    Raises:
      ValueError: If failed to create `ObjectDetector` object from the provided
        file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = ObjectDetectorOptions(
        base_options=base_options, running_mode=_RunningMode.IMAGE
    )
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(
      cls, options: ObjectDetectorOptions
  ) -> 'ObjectDetector':
    """Creates the `ObjectDetector` object from object detector options.

    Args:
      options: Options for the object detector task.

    Returns:
      `ObjectDetector` object that's created from `options`.

    Raises:
      ValueError: If failed to create `ObjectDetector` object from
        `ObjectDetectorOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """

    base_vision_task_api.validate_running_mode(
        options.running_mode, options.result_callback
    )

    lib = mediapipe_c_bindings_c_module.load_shared_library(_CTYPES_SIGNATURES)

    ctypes_options = options.to_ctypes()

    error_msg_ptr = ctypes.c_char_p()
    detector_handle = lib.object_detector_create(
        ctypes.byref(ctypes_options),
        ctypes.byref(error_msg_ptr),
    )

    if not detector_handle:
      if error_msg_ptr.value is not None:
        error_message = error_msg_ptr.value.decode('utf-8')
        raise RuntimeError(error_message)
      else:
        raise RuntimeError('Failed to create ObjectDetector object.')

    return ObjectDetector(lib=lib, handle=detector_handle)

  # TODO: Create an Image class for MediaPipe Tasks.
  def detect(
      self,
      image: image_module.Image,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> ObjectDetectorResult:
    """Performs object detection on the provided MediaPipe Image.

    Only use this method when the ObjectDetector is created with the image
    running mode.

    Args:
      image: MediaPipe Image.
      image_processing_options: Options for image processing.

    Returns:
      A detection result object that contains a list of detections, each
      detection has a bounding box that is expressed in the unrotated input
      frame of reference coordinates system, i.e. in `[0,image_width) x [0,
      image_height)`, which are the dimensions of the underlying image data.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If object detection failed to run.
    """

    c_image = image._image_ptr  # pylint: disable=protected-access
    c_result = detections_c_module.DetectionResultC()
    error_msg_ptr = ctypes.c_char_p()

    if image_processing_options:
      c_image_processing_options = image_processing_options.to_ctypes()
      status = self._lib.object_detector_detect_image_with_options(
          self._handle,
          c_image,
          ctypes.byref(c_image_processing_options),
          ctypes.byref(c_result),
          ctypes.byref(error_msg_ptr),
      )
    else:
      status = self._lib.object_detector_detect_image(
          self._handle,
          c_image,
          ctypes.byref(c_result),
          ctypes.byref(error_msg_ptr),
      )

    self._handle_status(
        status, error_msg_ptr, 'Failed to detect objects from image.'
    )

    py_result = ObjectDetectorResult.from_ctypes(c_result)
    self._lib.object_detector_close_result(ctypes.byref(c_result))
    return py_result

  def detect_for_video(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> ObjectDetectorResult:
    """Performs object detection on the provided video frames.

    Only use this method when the ObjectDetector is created with the video
    running mode. It's required to provide the video frame's timestamp (in
    milliseconds) along with the video frame. The input timestamps should be
    monotonically increasing for adjacent calls of this method.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input video frame in milliseconds.
      image_processing_options: Options for image processing.

    Returns:
      A detection result object that contains a list of detections, each
      detection has a bounding box that is expressed in the unrotated input
      frame of reference coordinates system, i.e. in `[0,image_width) x [0,
      image_height)`, which are the dimensions of the underlying image data.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If object detection failed to run.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    c_result = detections_c_module.DetectionResultC()
    error_msg_ptr = ctypes.c_char_p()

    if image_processing_options:
      c_image_processing_options = image_processing_options.to_ctypes()
      status = self._lib.object_detector_detect_for_video_with_options(
          self._handle,
          c_image,
          ctypes.byref(c_image_processing_options),
          timestamp_ms,
          ctypes.byref(c_result),
          ctypes.byref(error_msg_ptr),
      )
    else:
      status = self._lib.object_detector_detect_for_video(
          self._handle,
          c_image,
          timestamp_ms,
          ctypes.byref(c_result),
          ctypes.byref(error_msg_ptr),
      )

    self._handle_status(
        status, error_msg_ptr, 'Failed to detect objects from video.'
    )

    py_result = ObjectDetectorResult.from_ctypes(c_result)
    self._lib.object_detector_close_result(ctypes.byref(c_result))
    return py_result

  def detect_async(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> None:
    """Sends live image data (an Image with a unique timestamp) to perform object detection.

    Only use this method when the ObjectDetector is created with the live stream
    running mode. The input timestamps should be monotonically increasing for
    adjacent calls of this method. This method will return immediately after the
    input image is accepted. The results will be available via the
    `result_callback` provided in the `ObjectDetectorOptions`. The
    `detect_async` method is designed to process live stream data such as camera
    input. To lower the overall latency, object detector may drop the input
    images if needed. In other words, it's not guaranteed to have output per
    input image.

    The `result_callback` provides:
      - A detection result object that contains a list of detections, each
        detection has a bounding box that is expressed in the unrotated input
        frame of reference coordinates system, i.e. in `[0,image_width) x [0,
        image_height)`, which are the dimensions of the underlying image data.
      - The input image that the object detector runs on.
      - The input timestamp in milliseconds.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input image in milliseconds.
      image_processing_options: Options for image processing.

    Raises:
      ValueError: If the current input timestamp is smaller than what the object
        detector has already processed.
      RuntimeError: If object detection failed to run.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    error_msg_ptr = ctypes.c_char_p()

    if image_processing_options:
      c_image_processing_options = image_processing_options.to_ctypes()
      status = self._lib.object_detector_detect_async_with_options(
          self._handle,
          c_image,
          ctypes.byref(c_image_processing_options),
          timestamp_ms,
          ctypes.byref(error_msg_ptr),
      )
    else:
      status = self._lib.object_detector_detect_async(
          self._handle,
          c_image,
          timestamp_ms,
          ctypes.byref(error_msg_ptr),
      )

    self._handle_status(
        status, error_msg_ptr, 'Failed to detect objects asynchronously.'
    )

  def close(self):
    """Shuts down the MediaPipe task instance."""
    if self._handle:
      error_msg_ptr = ctypes.c_char_p()
      ret_code = self._lib.object_detector_close(
          self._handle, ctypes.byref(error_msg_ptr)
      )
      self._handle_status(
          ret_code, error_msg_ptr, 'Failed to close ObjectDetector object.'
      )
      self._handle = None
      self._lib.close()

  def _handle_status(
      self, status: int, error_msg_ptr: ctypes.c_char_p, default_error_msg: str
  ):
    if status != 0:
      if error_msg_ptr.value is not None:
        error_message = error_msg_ptr.value.decode('utf-8')
        raise RuntimeError(error_message)
      else:
        raise RuntimeError(default_error_msg)

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
