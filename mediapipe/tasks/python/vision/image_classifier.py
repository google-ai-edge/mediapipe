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
"""MediaPipe image classifier task."""

import ctypes
import dataclasses
from typing import Callable, List, Optional, Tuple

from mediapipe.tasks.python.components.containers import classification_result as classification_result_module
from mediapipe.tasks.python.components.containers import classification_result_c
from mediapipe.tasks.python.components.processors import classifier_options as classifier_options_module
from mediapipe.tasks.python.components.processors import classifier_options_c
from mediapipe.tasks.python.core import async_result_dispatcher
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import base_options_c
from mediapipe.tasks.python.core import mediapipe_c_bindings
from mediapipe.tasks.python.core import mediapipe_c_utils
from mediapipe.tasks.python.core import serial_dispatcher
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import image as image_lib
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import image_processing_options_c
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

ImageClassifierResult = classification_result_module.ClassificationResult
_BaseOptions = base_options_module.BaseOptions
_ClassifierOptions = classifier_options_module.ClassifierOptions
_RunningMode = running_mode_module.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions
_AsyncResultDispatcher = async_result_dispatcher.AsyncResultDispatcher


_C_TYPES_RESULT_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int32,  # MpStatus
    ctypes.POINTER(classification_result_c.ClassificationResultC),
    ctypes.c_void_p,  # MpImage
    ctypes.c_int64,  # timestamp_ms
)


class ImageClassifierOptionsC(ctypes.Structure):
  """The ctypes struct for ImageClassifierOptions."""

  _fields_ = [
      ("base_options", base_options_c.BaseOptionsC),
      ("running_mode", ctypes.c_int),
      (
          "classifier_options",
          classifier_options_c.ClassifierOptionsC,
      ),
      ("result_callback", _C_TYPES_RESULT_CALLBACK),
  ]

  @classmethod
  @doc_controls.do_not_generate_docs
  def from_c_options(
      cls,
      base_options: base_options_c.BaseOptionsC,
      running_mode: _RunningMode,
      classifier_options: classifier_options_c.ClassifierOptionsC,
      result_callback: _C_TYPES_RESULT_CALLBACK,
  ) -> "ImageClassifierOptionsC":
    """Creates an ImageClassifierOptionsC object from the given options."""
    return cls(
        base_options=base_options,
        running_mode=running_mode.ctype,
        classifier_options=classifier_options,
        result_callback=result_callback,
    )


_CTYPES_SIGNATURES = (
    mediapipe_c_utils.CStatusFunction(
        "MpImageClassifierCreate",
        (
            ctypes.POINTER(ImageClassifierOptionsC),
            ctypes.POINTER(ctypes.c_void_p),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        "MpImageClassifierClassifyImage",
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(image_processing_options_c.ImageProcessingOptionsC),
            ctypes.POINTER(classification_result_c.ClassificationResultC),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        "MpImageClassifierClassifyForVideo",
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(image_processing_options_c.ImageProcessingOptionsC),
            ctypes.c_int64,
            ctypes.POINTER(classification_result_c.ClassificationResultC),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        "MpImageClassifierClassifyAsync",
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(image_processing_options_c.ImageProcessingOptionsC),
            ctypes.c_int64,
        ),
    ),
    mediapipe_c_utils.CFunction(
        "MpImageClassifierCloseResult",
        [ctypes.POINTER(classification_result_c.ClassificationResultC)],
        None,
    ),
    mediapipe_c_utils.CStatusFunction(
        "MpImageClassifierClose",
        (ctypes.c_void_p,),
    ),
)


@dataclasses.dataclass
class ImageClassifierOptions:
  """Options for the image classifier task.

  Attributes:
    base_options: Base options for the image classifier task.
    running_mode: The running mode of the task. Default to the image mode. Image
      classifier task has three running modes: 1) The image mode for classifying
      objects on single image inputs. 2) The video mode for classifying objects
      on the decoded frames of a video. 3) The live stream mode for classifying
      objects on a live stream of input data, such as from camera.
    display_names_locale: The locale to use for display names specified through
      the TFLite Model Metadata.
    max_results: The maximum number of top-scored classification results to
      return.
    score_threshold: Overrides the ones provided in the model metadata. Results
      below this value are rejected.
    category_allowlist: Allowlist of category names. If non-empty,
      classification results whose category name is not in this set will be
      filtered out. Duplicate or unknown category names are ignored. Mutually
      exclusive with `category_denylist`.
    category_denylist: Denylist of category names. If non-empty, classification
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
  max_results: Optional[int] = None
  score_threshold: Optional[float] = None
  category_allowlist: Optional[List[str]] = None
  category_denylist: Optional[List[str]] = None
  result_callback: Optional[
      Callable[[ImageClassifierResult, image_lib.Image, int], None]
  ] = None


class ImageClassifier:
  """Class that performs image classification on images.

  The API expects a TFLite model with optional, but strongly recommended,
  TFLite Model Metadata.

  Input tensor:
    (kTfLiteUInt8/kTfLiteFloat32)
    - image input of size `[batch x height x width x channels]`.
    - batch inference is not supported (`batch` is required to be 1).
    - only RGB inputs are supported (`channels` is required to be 3).
    - if type is kTfLiteFloat32, NormalizationOptions are required to be
      attached to the metadata for input normalization.
  At least one output tensor with:
    (kTfLiteUInt8/kTfLiteFloat32)
    - `N `classes and either 2 or 4 dimensions, i.e. `[1 x N]` or
      `[1 x 1 x 1 x N]`
    - optional (but recommended) label map(s) as AssociatedFiles with type
      TENSOR_AXIS_LABELS, containing one label per line. The first such
      AssociatedFile (if any) is used to fill the `class_name` field of the
      results. The `display_name` field is filled from the AssociatedFile (if
      any) whose locale matches the `display_names_locale` field of the
      `ImageClassifierOptions` used at creation time ("en" by default, i.e.
      English). If none of these are available, only the `index` field of the
      results will be filled.
    - optional score calibration can be attached using ScoreCalibrationOptions
      and an AssociatedFile with type TENSOR_AXIS_SCORE_CALIBRATION. See
      metadata_schema.fbs [1] for more details.

  An example of such model can be found at:
  https://tfhub.dev/bohemian-visual-recognition-alliance/lite-model/models/mushroom-identification_v1/1

  [1]:
  https://github.com/google/mediapipe/blob/6cdc6443b6a7ed662744e2a2ce2d58d9c83e6d6f/mediapipe/tasks/metadata/metadata_schema.fbs#L456
  """

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
    """Initializes the `ImageClassifier` object.

    Args:
      lib: The dispatch library to use for the image classifier.
      handle: The C pointer to the image classifier.
      dispatcher: The async result handler for the image classifier.
      async_callback: The c callback for the image classifier.
    """
    self._lib = lib
    self._handle = handle
    self._dispatcher = dispatcher
    self._async_callback = async_callback

  @classmethod
  def create_from_model_path(cls, model_path: str) -> "ImageClassifier":
    """Creates an `ImageClassifier` object from a TensorFlow Lite model and the default `ImageClassifierOptions`.

    Note that the created `ImageClassifier` instance is in image mode, for
    classifying objects on single image inputs.

    Args:
      model_path: Path to the model.

    Returns:
      `ImageClassifier` object that's created from the model file and the
      default `ImageClassifierOptions`.

    Raises:
      ValueError: If failed to create `ImageClassifier` object from the provided
        file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = ImageClassifierOptions(
        base_options=base_options, running_mode=_RunningMode.IMAGE
    )
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(
      cls, options: ImageClassifierOptions
  ) -> "ImageClassifier":
    """Creates the `ImageClassifier` object from image classifier options.

    Args:
      options: Options for the image classifier task.

    Returns:
      `ImageClassifier` object that's created from `options`.

    Raises:
      ValueError: If failed to create `ImageClassifier` object from
        `ImageClassifierOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """
    running_mode_module.validate_running_mode(
        options.running_mode, options.result_callback
    )

    lib = mediapipe_c_bindings.load_shared_library(_CTYPES_SIGNATURES)

    def convert_result(
        c_result_ptr: ctypes.POINTER(
            classification_result_c.ClassificationResultC
        ),
        image_ptr: ctypes.c_void_p,
        timestamp_ms: int,
    ) -> Tuple[ImageClassifierResult, image_lib.Image, int]:
      c_result = c_result_ptr[0]
      py_result = ImageClassifierResult.from_ctypes(c_result)
      py_image = image_lib.Image.create_from_ctypes(image_ptr)
      return (py_result, py_image, timestamp_ms)

    dispatcher = _AsyncResultDispatcher(converter=convert_result)
    c_callback = dispatcher.wrap_callback(
        options.result_callback, _C_TYPES_RESULT_CALLBACK
    )
    options_c = ImageClassifierOptionsC.from_c_options(
        base_options=options.base_options.to_ctypes(),
        running_mode=options.running_mode,
        classifier_options=classifier_options_c.convert_to_classifier_options_c(
            _ClassifierOptions(
                max_results=options.max_results,
                score_threshold=options.score_threshold,
                category_allowlist=options.category_allowlist,
                category_denylist=options.category_denylist,
                display_names_locale=options.display_names_locale,
            )
        ),
        result_callback=c_callback,
    )
    classifier_handle = ctypes.c_void_p()
    lib.MpImageClassifierCreate(
        ctypes.byref(options_c), ctypes.byref(classifier_handle)
    )
    return cls(
        lib=lib,
        handle=classifier_handle,
        dispatcher=dispatcher,
        async_callback=c_callback,
    )

  def classify(
      self,
      image: image_lib.Image,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> ImageClassifierResult:
    """Performs image classification on the provided MediaPipe Image.

    Only use this method when the ImageClassifier is created with the image
    running mode.

    Args:
      image: MediaPipe Image.
      image_processing_options: Options for image processing.

    Returns:
      A classification result object that contains a list of classifications.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If image classification failed to run.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    c_result = classification_result_c.ClassificationResultC()
    options_c = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpImageClassifierClassifyImage(
        self._handle,
        c_image,
        options_c,
        ctypes.byref(c_result),
    )

    result = ImageClassifierResult.from_ctypes(c_result)
    self._lib.MpImageClassifierCloseResult(ctypes.byref(c_result))
    return result

  def classify_for_video(
      self,
      image: image_lib.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> ImageClassifierResult:
    """Performs image classification on the provided video frames.

    Only use this method when the ImageClassifier is created with the video
    running mode. It's required to provide the video frame's timestamp (in
    milliseconds) along with the video frame. The input timestamps should be
    monotonically increasing for adjacent calls of this method.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input video frame in milliseconds.
      image_processing_options: Options for image processing.

    Returns:
      A classification result object that contains a list of classifications.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If image classification failed to run.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    c_result = classification_result_c.ClassificationResultC()
    options_c = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpImageClassifierClassifyForVideo(
        self._handle,
        c_image,
        options_c,
        timestamp_ms,
        ctypes.byref(c_result),
    )

    result = ImageClassifierResult.from_ctypes(c_result)
    self._lib.MpImageClassifierCloseResult(ctypes.byref(c_result))
    return result

  def classify_async(
      self,
      image: image_lib.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> None:
    """Sends live image data (an Image with a unique timestamp) to perform image classification.

    Only use this method when the ImageClassifier is created with the live
    stream running mode. The input timestamps should be monotonically increasing
    for adjacent calls of this method. This method will return immediately after
    the input image is accepted. The results will be available via the
    `result_callback` provided in the `ImageClassifierOptions`. The
    `classify_async` method is designed to process live stream data such as
    camera input. To lower the overall latency, image classifier may drop the
    input images if needed. In other words, it's not guaranteed to have output
    per input image.

    The `result_callback` provides:
      - A classification result object that contains a list of classifications.
      - The input image that the image classifier runs on.
      - The input timestamp in milliseconds.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input image in milliseconds.
      image_processing_options: Options for image processing.

    Raises:
      ValueError: If the current input timestamp is smaller than what the image
        classifier has already processed.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    options_c = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpImageClassifierClassifyAsync(
        self._handle,
        c_image,
        options_c,
        timestamp_ms,
    )

  def close(self):
    """Closes ImageClassifier."""
    if not self._handle:
      return
    self._lib.MpImageClassifierClose(self._handle)
    self._handle = None
    self._dispatcher.close()
    self._lib.close()

  def __enter__(self):
    """Returns `self` upon entering the runtime context."""
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Closes ImageClassifier and exits the context manager.

    Args:
      exc_type: The exception type that caused the context manager to exit.
      exc_value: The exception value that caused the context manager to exit.
      traceback: The exception traceback that caused the context manager to
        exit.

    Raises:
      RuntimeError: If the MediaPipe Image Classifier task failed to
      close.
    """
    self.close()

  def __del__(self):
    self.close()
