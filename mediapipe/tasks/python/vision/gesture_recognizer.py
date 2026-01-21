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
"""MediaPipe gesture recognizer task."""

import ctypes
import dataclasses
from typing import Callable, Optional, Tuple

from mediapipe.tasks.python.components.processors import classifier_options
from mediapipe.tasks.python.components.processors import classifier_options_c
from mediapipe.tasks.python.core import async_result_dispatcher
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import base_options_c
from mediapipe.tasks.python.core import mediapipe_c_bindings
from mediapipe.tasks.python.core import mediapipe_c_utils
from mediapipe.tasks.python.core import serial_dispatcher
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision import gesture_recognizer_result as gesture_recognizer_result_module
# C-bindings
from mediapipe.tasks.python.vision import gesture_recognizer_result_c
from mediapipe.tasks.python.vision.core import image as image_lib
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_lib
from mediapipe.tasks.python.vision.core import image_processing_options_c
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

_BaseOptions = base_options_module.BaseOptions
_ClassifierOptions = classifier_options.ClassifierOptions
_RunningMode = running_mode_module.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_lib.ImageProcessingOptions
GestureRecognizerResult = (
    gesture_recognizer_result_module.GestureRecognizerResult
)


_C_TYPES_RESULT_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int32,  # MpStatus
    ctypes.POINTER(gesture_recognizer_result_c.GestureRecognizerResultC),
    ctypes.c_void_p,  # MpImage
    ctypes.c_int64,  # timestamp_ms
)


class GestureRecognizerOptionsC(ctypes.Structure):
  """C types for GestureRecognizerOptions."""
  _fields_ = [
      ('base_options', base_options_c.BaseOptionsC),
      ('running_mode', ctypes.c_int),
      ('num_hands', ctypes.c_int),
      ('min_hand_detection_confidence', ctypes.c_float),
      ('min_hand_presence_confidence', ctypes.c_float),
      ('min_tracking_confidence', ctypes.c_float),
      (
          'canned_gestures_classifier_options',
          classifier_options_c.ClassifierOptionsC,
      ),
      (
          'custom_gestures_classifier_options',
          classifier_options_c.ClassifierOptionsC,
      ),
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
      canned_gestures_classifier_options: classifier_options_c.ClassifierOptionsC,
      custom_gestures_classifier_options: classifier_options_c.ClassifierOptionsC,
      result_callback: '_C_TYPES_RESULT_CALLBACK',
  ) -> 'GestureRecognizerOptionsC':
    """Creates a GestureRecognizerOptionsC object from the given options."""
    return cls(
        base_options=base_options,
        running_mode=running_mode.ctype,
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        canned_gestures_classifier_options=canned_gestures_classifier_options,
        custom_gestures_classifier_options=custom_gestures_classifier_options,
        result_callback=result_callback,
    )


_CTYPES_SIGNATURES = (
    mediapipe_c_utils.CStatusFunction(
        'MpGestureRecognizerCreate',
        (
            ctypes.POINTER(GestureRecognizerOptionsC),
            ctypes.POINTER(ctypes.c_void_p),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpGestureRecognizerRecognizeImage',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(image_processing_options_c.ImageProcessingOptionsC),
            ctypes.POINTER(
                gesture_recognizer_result_c.GestureRecognizerResultC
            ),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpGestureRecognizerRecognizeForVideo',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(image_processing_options_c.ImageProcessingOptionsC),
            ctypes.c_int64,
            ctypes.POINTER(
                gesture_recognizer_result_c.GestureRecognizerResultC
            ),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpGestureRecognizerRecognizeAsync',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(image_processing_options_c.ImageProcessingOptionsC),
            ctypes.c_int64,
        ),
    ),
    mediapipe_c_utils.CFunction(
        'MpGestureRecognizerCloseResult',
        [ctypes.POINTER(gesture_recognizer_result_c.GestureRecognizerResultC)],
        None,
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpGestureRecognizerClose',
        (ctypes.c_void_p,),
    ),
)


@dataclasses.dataclass
class GestureRecognizerOptions:
  """Options for the gesture recognizer task.

  Attributes:
    base_options: Base options for the hand gesture recognizer task.
    running_mode: The running mode of the task. Default to the image mode.
      Gesture recognizer task has three running modes: 1) The image mode for
      recognizing hand gestures on single image inputs. 2) The video mode for
      recognizing hand gestures on the decoded frames of a video. 3) The live
      stream mode for recognizing hand gestures on a live stream of input data,
      such as from camera.
    num_hands: The maximum number of hands can be detected by the recognizer.
    min_hand_detection_confidence: The minimum confidence score for the hand
      detection to be considered successful.
    min_hand_presence_confidence: The minimum confidence score of hand presence
      score in the hand landmark detection.
    min_tracking_confidence: The minimum confidence score for the hand tracking
      to be considered successful.
    canned_gesture_classifier_options: Options for configuring the canned
      gestures classifier, such as score threshold, allow list and deny list of
      gestures. The categories for canned gesture classifiers are: ["None",
      "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up",
      "Victory", "ILoveYou"]. Note this option is subject to change.
    custom_gesture_classifier_options: Options for configuring the custom
      gestures classifier, such as score threshold, allow list and deny list of
      gestures. Note this option is subject to change.
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
  canned_gesture_classifier_options: _ClassifierOptions = dataclasses.field(
      default_factory=_ClassifierOptions
  )
  custom_gesture_classifier_options: _ClassifierOptions = dataclasses.field(
      default_factory=_ClassifierOptions
  )
  result_callback: Optional[
      Callable[[GestureRecognizerResult, image_lib.Image, int], None]
  ] = None


class GestureRecognizer:
  """Class that performs gesture recognition on images."""

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
    """Initializes the gesture recognizer.

    Args:
      lib: The dispatch library to use for the gesture recognizer.
      handle: The C pointer to the gesture recognizer.
      dispatcher: The async result handler for the gesture recognizer.
      async_callback: The c callback for the gesture recognizer.
    """
    self._lib = lib
    self._handle = handle
    self._dispatcher = dispatcher
    self._async_callback = async_callback

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'GestureRecognizer':
    """Creates an `GestureRecognizer` object from a TensorFlow Lite model and the default `GestureRecognizerOptions`.

    Note that the created `GestureRecognizer` instance is in image mode, for
    recognizing hand gestures on single image inputs.

    Args:
      model_path: Path to the model.

    Returns:
      `GestureRecognizer` object that's created from the model file and the
      default `GestureRecognizerOptions`.

    Raises:
      ValueError: If failed to create `GestureRecognizer` object from the
        provided file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = GestureRecognizerOptions(
        base_options=base_options, running_mode=_RunningMode.IMAGE
    )
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(
      cls, options: GestureRecognizerOptions
  ) -> 'GestureRecognizer':
    """Creates the `GestureRecognizer` object from gesture recognizer options.

    Args:
      options: Options for the gesture recognizer task.

    Returns:
      `GestureRecognizer` object that's created from `options`.

    Raises:
      ValueError: If failed to create `GestureRecognizer` object from
        `GestureRecognizerOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """
    running_mode_module.validate_running_mode(
        options.running_mode, options.result_callback
    )

    lib = mediapipe_c_bindings.load_shared_library(_CTYPES_SIGNATURES)

    def convert_result(
        c_result_ptr: ctypes.POINTER(
            gesture_recognizer_result_c.GestureRecognizerResultC
        ),
        image_ptr: ctypes.c_void_p,
        timestamp_ms: int,
    ) -> Tuple[GestureRecognizerResult, image_lib.Image, int]:
      c_result = c_result_ptr[0]
      py_result = GestureRecognizerResult.from_ctypes(c_result)
      py_image = image_lib.Image.create_from_ctypes(image_ptr)
      return (py_result, py_image, timestamp_ms)

    dispatcher = async_result_dispatcher.AsyncResultDispatcher(
        converter=convert_result
    )
    c_callback = dispatcher.wrap_callback(
        options.result_callback, _C_TYPES_RESULT_CALLBACK
    )
    options_c = GestureRecognizerOptionsC.from_c_options(
        base_options=options.base_options.to_ctypes(),
        running_mode=options.running_mode,
        num_hands=options.num_hands,
        min_hand_detection_confidence=options.min_hand_detection_confidence,
        min_hand_presence_confidence=options.min_hand_presence_confidence,
        min_tracking_confidence=options.min_tracking_confidence,
        canned_gestures_classifier_options=classifier_options_c.convert_to_classifier_options_c(
            options.canned_gesture_classifier_options
        ),
        custom_gestures_classifier_options=classifier_options_c.convert_to_classifier_options_c(
            options.custom_gesture_classifier_options
        ),
        result_callback=c_callback,
    )
    recognizer_handle = ctypes.c_void_p()
    lib.MpGestureRecognizerCreate(
        ctypes.byref(options_c),
        ctypes.byref(recognizer_handle)
    )
    return cls(
        lib=lib,
        handle=recognizer_handle,
        dispatcher=dispatcher,
        async_callback=c_callback,
    )

  def recognize(
      self,
      image: image_lib.Image,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> GestureRecognizerResult:
    """Performs hand gesture recognition on the given image.

    Only use this method when the GestureRecognizer is created with the image
    running mode.

    The image can be of any size with format RGB or RGBA.
    TODO: Describes how the input image will be preprocessed after the yuv
    support is implemented.

    Args:
      image: MediaPipe Image.
      image_processing_options: Options for image processing.

    Returns:
      The hand gesture recognition results.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If gesture recognition failed to run.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    c_result = gesture_recognizer_result_c.GestureRecognizerResultC()
    options_c = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpGestureRecognizerRecognizeImage(
        self._handle,
        c_image,
        options_c,
        ctypes.byref(c_result),
    )

    result = GestureRecognizerResult.from_ctypes(c_result)
    self._lib.MpGestureRecognizerCloseResult(ctypes.byref(c_result))
    return result

  def recognize_for_video(
      self,
      image: image_lib.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> GestureRecognizerResult:
    """Performs gesture recognition on the provided video frame.

    Only use this method when the GestureRecognizer is created with the video
    running mode.

    Only use this method when the GestureRecognizer is created with the video
    running mode. It's required to provide the video frame's timestamp (in
    milliseconds) along with the video frame. The input timestamps should be
    monotonically increasing for adjacent calls of this method.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input video frame in milliseconds.
      image_processing_options: Options for image processing.

    Returns:
      The hand gesture recognition results.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If gesture recognition failed to run.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    c_result = gesture_recognizer_result_c.GestureRecognizerResultC()
    options_c = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpGestureRecognizerRecognizeForVideo(
        self._handle,
        c_image,
        options_c,
        timestamp_ms,
        ctypes.byref(c_result),
    )

    result = GestureRecognizerResult.from_ctypes(c_result)
    self._lib.MpGestureRecognizerCloseResult(ctypes.byref(c_result))
    return result

  def recognize_async(
      self,
      image: image_lib.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> None:
    """Sends live image data to perform gesture recognition.

    The results will be available via the "result_callback" provided in the
    GestureRecognizerOptions. Only use this method when the GestureRecognizer
    is created with the live stream running mode.

    Only use this method when the GestureRecognizer is created with the live
    stream running mode. The input timestamps should be monotonically increasing
    for adjacent calls of this method. This method will return immediately after
    the input image is accepted. The results will be available via the
    `result_callback` provided in the `GestureRecognizerOptions`. The
    `recognize_async` method is designed to process live stream data such as
    camera input. To lower the overall latency, gesture recognizer may drop the
    input images if needed. In other words, it's not guaranteed to have output
    per input image.

    The `result_callback` provides:
      - The hand gesture recognition results.
      - The input image that the gesture recognizer runs on.
      - The input timestamp in milliseconds.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input image in milliseconds.
      image_processing_options: Options for image processing.

    Raises:
      ValueError: If the current input timestamp is smaller than what the
      gesture recognizer has already processed.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    options_c = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpGestureRecognizerRecognizeAsync(
        self._handle,
        c_image,
        options_c,
        timestamp_ms,
    )

  def close(self):
    """Closes GestureRecognizer."""
    if not self._handle:
      return
    self._lib.MpGestureRecognizerClose(self._handle)
    self._handle = None
    self._dispatcher.close()
    self._lib.close()

  def __enter__(self):
    """Returns `self` upon entering the runtime context."""
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Closes GestureRecognizers and exits the context manager.

    Args:
      exc_type: The exception type that caused the context manager to exit.
      exc_value: The exception value that caused the context manager to exit.
      traceback: The exception traceback that caused the context manager to
        exit.

    Raises:
      RuntimeError: If the MediaPipe Gesture Recognizer task failed to
      close.
    """
    self.close()

  def __del__(self):
    self.close()
