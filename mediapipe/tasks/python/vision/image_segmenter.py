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
"""MediaPipe image segmenter task."""

import ctypes
import dataclasses
import logging
from typing import Callable, Optional

from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import base_options_c
from mediapipe.tasks.python.core import mediapipe_c_bindings
from mediapipe.tasks.python.core import serial_dispatcher
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import base_vision_task_api
from mediapipe.tasks.python.vision.core import image as image_module
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import image_processing_options_c
from mediapipe.tasks.python.vision.core import vision_task_running_mode

_BaseOptions = base_options_module.BaseOptions
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions
_RunningMode = vision_task_running_mode.VisionTaskRunningMode
_CFunction = mediapipe_c_bindings.CFunction


class MpStringListC(ctypes.Structure):
  _fields_ = [
      ('strings', ctypes.POINTER(ctypes.c_char_p)),
      ('num_strings', ctypes.c_int),
  ]


class ImageSegmenterResultC(ctypes.Structure):
  _fields_ = [
      ('confidence_masks', ctypes.POINTER(ctypes.c_void_p)),
      ('confidence_masks_count', ctypes.c_uint32),
      ('has_confidence_masks', ctypes.c_uint32),
      ('category_mask', ctypes.c_void_p),
      ('has_category_mask', ctypes.c_uint32),
      ('quality_scores', ctypes.POINTER(ctypes.c_float)),
      ('quality_scores_count', ctypes.c_uint32),
  ]


class ImageSegmenterOptionsC(ctypes.Structure):
  _fields_ = [
      ('base_options', base_options_c.BaseOptionsC),
      ('running_mode', ctypes.c_int),
      ('display_names_locale', ctypes.c_char_p),
      ('output_confidence_masks', ctypes.c_bool),
      ('output_category_mask', ctypes.c_bool),
      (
          'result_callback',
          ctypes.CFUNCTYPE(
              None,
              ctypes.POINTER(ImageSegmenterResultC),
              ctypes.c_void_p,
              ctypes.c_int64,
              ctypes.c_char_p,
          ),
      ),
  ]


_CTYPES_SIGNATURES = (
    _CFunction(
        'image_segmenter_create',
        [
            ctypes.POINTER(ImageSegmenterOptionsC),
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_void_p,
    ),
    _CFunction(
        'image_segmenter_segment_image_with_options',
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(image_processing_options_c.ImageProcessingOptionsC),
            ctypes.POINTER(ImageSegmenterResultC),
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_int,
    ),
    _CFunction(
        'image_segmenter_segment_for_video_with_options',
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(image_processing_options_c.ImageProcessingOptionsC),
            ctypes.c_int64,
            ctypes.POINTER(ImageSegmenterResultC),
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_int,
    ),
    _CFunction(
        'image_segmenter_segment_async_with_options',
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(image_processing_options_c.ImageProcessingOptionsC),
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_int,
    ),
    _CFunction(
        'image_segmenter_get_labels',
        [
            ctypes.c_void_p,
            ctypes.POINTER(MpStringListC),
        ],
        ctypes.c_int,
    ),
    _CFunction(
        'image_segmenter_close_result',
        [ctypes.POINTER(ImageSegmenterResultC)],
        None,
    ),
    _CFunction(
        'image_segmenter_close',
        [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char_p),
        ],
        ctypes.c_int,
    ),
    _CFunction(
        'MpStringListFree',
        [
            ctypes.c_void_p,
        ],
        None,
    ),
)


@dataclasses.dataclass
class ImageSegmenterResult:
  """Output result of ImageSegmenter.

  confidence_masks: multiple masks of float image where, for each mask, each
  pixel represents the prediction confidence, usually in the [0, 1] range.

  category_mask: a category mask of uint8 image where each pixel represents the
  class which the pixel in the original image was predicted to belong to.
  """

  confidence_masks: Optional[list[image_module.Image]] = None
  category_mask: Optional[image_module.Image] = None

  @classmethod
  @doc_controls.do_not_generate_docs
  def from_ctypes(
      cls, c_result: ImageSegmenterResultC
  ) -> 'ImageSegmenterResult':
    """Creates an `ImageSegmenterResult` object from the given ctypes struct."""
    confidence_masks = None
    category_mask = None
    if (
        c_result.has_confidence_masks > 0
        and c_result.confidence_masks_count > 0
    ):
      confidence_masks = [
          image_module.Image.create_from_ctypes(c_result.confidence_masks[i])
          for i in range(c_result.confidence_masks_count)
      ]
    if c_result.has_category_mask > 0:
      category_mask = image_module.Image.create_from_ctypes(
          c_result.category_mask
      )

    return cls(
        confidence_masks=confidence_masks,
        category_mask=category_mask,
    )


@dataclasses.dataclass
class ImageSegmenterOptions:
  """Options for the image segmenter task.

  Attributes:
    base_options: Base options for the image segmenter task.
    running_mode: The running mode of the task. Default to the image mode. Image
      segmenter task has three running modes: 1) The image mode for segmenting
      objects on single image inputs. 2) The video mode for segmenting objects
      on the decoded frames of a video. 3) The live stream mode for segmenting
      objects on a live stream of input data, such as from camera.
    output_confidence_masks: Whether to output confidence masks.
    output_category_mask: Whether to output category mask.
    result_callback: The user-defined result callback for processing live stream
      data. The result callback should only be specified when the running mode
      is set to the live stream mode.
  """

  base_options: _BaseOptions
  running_mode: _RunningMode = _RunningMode.IMAGE
  output_confidence_masks: bool = True
  output_category_mask: bool = False
  result_callback: (
      Callable[[ImageSegmenterResult, image_module.Image, int], None] | None
  ) = None

  _result_callback_c: (
      Callable[
          [
              ImageSegmenterResultC,
              ctypes.c_void_p,
              int,
              str,
          ],
          None,
      ]
      | None
  ) = None

  @doc_controls.do_not_generate_docs
  def to_ctypes(self) -> ImageSegmenterOptionsC:
    """Generates an ImageSegmenterOptionsC ctypes struct."""
    if not self._result_callback_c:
      result_callback_fn = ctypes.CFUNCTYPE(
          None,
          ctypes.POINTER(ImageSegmenterResultC),
          ctypes.c_void_p,
          ctypes.c_int64,
          ctypes.c_char_p,
      )

      @result_callback_fn
      def c_callback(result, image, timestamp_ms, error_msg):
        if self.result_callback:
          if error_msg:
            logging.error('Image segmenter error: %s', error_msg.decode())
            return
          py_result = ImageSegmenterResult.from_ctypes(result.contents)
          py_image = image_module.Image.create_from_ctypes(image)
          self.result_callback(py_result, py_image, timestamp_ms)

      self._result_callback_c = c_callback

    empty_string = ctypes.c_char_p(b'')
    return ImageSegmenterOptionsC(
        base_options=self.base_options.to_ctypes(),
        running_mode=self.running_mode.ctype,
        display_names_locale=empty_string,
        output_confidence_masks=self.output_confidence_masks,
        output_category_mask=self.output_category_mask,
        result_callback=self._result_callback_c,
    )


class ImageSegmenter:
  """Class that performs image segmentation on images.

  The API expects a TFLite model with mandatory TFLite Model Metadata.

  Input tensor:
    (kTfLiteUInt8/kTfLiteFloat32)
    - image input of size `[batch x height x width x channels]`.
    - batch inference is not supported (`batch` is required to be 1).
    - RGB and greyscale inputs are supported (`channels` is required to be
      1 or 3).
    - if type is kTfLiteFloat32, NormalizationOptions are required to be
      attached to the metadata for input normalization.
  Output tensors:
    (kTfLiteUInt8/kTfLiteFloat32)
    - list of segmented masks.
    - if `output_category_mask` is True, uint8 Image, Image vector of size 1.
    - if `output_confidence_masks` is True, float32 Image list of size
      `channels`.
    - batch is always 1

  An example of such model can be found at:
  https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/metadata/2
  """

  _lib: serial_dispatcher.SerialDispatcher
  _handle: ctypes.c_void_p

  def __init__(
      self, lib: serial_dispatcher.SerialDispatcher, handle: ctypes.c_void_p
  ) -> None:
    """Initializes the `ImageSegmenter` object."""
    self._lib = lib
    self._handle = handle
    self._labels = None

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'ImageSegmenter':
    """Creates an `ImageSegmenter` object from a TensorFlow Lite model and the default `ImageSegmenterOptions`.

    Note that the created `ImageSegmenter` instance is in image mode, for
    performing image segmentation on single image inputs.

    Args:
      model_path: Path to the model.

    Returns:
      `ImageSegmenter` object that's created from the model file and the default
      `ImageSegmenterOptions`.

    Raises:
      ValueError: If failed to create `ImageSegmenter` object from the provided
        file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = ImageSegmenterOptions(
        base_options=base_options, running_mode=_RunningMode.IMAGE
    )
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(
      cls, options: ImageSegmenterOptions
  ) -> 'ImageSegmenter':
    """Creates the `ImageSegmenter` object from image segmenter options.

    Args:
      options: Options for the image segmenter task.

    Returns:
      `ImageSegmenter` object that's created from `options`.

    Raises:
      ValueError: If failed to create `ImageSegmenter` object from
        `ImageSegmenterOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """

    base_vision_task_api.validate_running_mode(
        options.running_mode, options.result_callback
    )

    lib = mediapipe_c_bindings.load_shared_library(_CTYPES_SIGNATURES)

    options_c = options.to_ctypes()
    error_msg_ptr = ctypes.c_char_p()
    segmenter_handle = lib.image_segmenter_create(
        ctypes.byref(options_c), ctypes.byref(error_msg_ptr)
    )

    if not segmenter_handle:
      if not error_msg_ptr.value:
        raise RuntimeError('Failed to create ImageSegmenter object.')
      error_message = error_msg_ptr.value.decode('utf-8')
      raise RuntimeError(error_message)
    return cls(lib=lib, handle=segmenter_handle)

  def segment(
      self,
      image: image_module.Image,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> ImageSegmenterResult:
    """Performs the actual segmentation task on the provided MediaPipe Image.

    Args:
      image: MediaPipe Image.
      image_processing_options: Options for image processing.

    Returns:
      If the output_type is CATEGORY_MASK, the returned vector of images is
      per-category segmented image mask.
      If the output_type is CONFIDENCE_MASK, the returned vector of images
      contains only one confidence image mask. A segmentation result object that
      contains a list of segmentation masks as images.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If image segmentation failed to run.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    c_result = ImageSegmenterResultC()
    error_msg = ctypes.c_char_p()
    options_c = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    status = self._lib.image_segmenter_segment_image_with_options(
        self._handle,
        c_image,
        options_c,
        ctypes.byref(c_result),
        ctypes.byref(error_msg),
    )
    mediapipe_c_bindings.handle_return_code(
        status, 'Failed to segment image', error_msg
    )
    result = ImageSegmenterResult.from_ctypes(c_result)
    self._lib.image_segmenter_close_result(ctypes.byref(c_result))
    return result

  def segment_for_video(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: _ImageProcessingOptions | None = None,
  ) -> ImageSegmenterResult:
    """Performs segmentation on the provided video frames.

    Only use this method when the ImageSegmenter is created with the video
    running mode. It's required to provide the video frame's timestamp (in
    milliseconds) along with the video frame. The input timestamps should be
    monotonically increasing for adjacent calls of this method.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input video frame in milliseconds.
      image_processing_options: Options for image processing.

    Returns:
      If the output_type is CATEGORY_MASK, the returned vector of images is
      per-category segmented image mask.
      If the output_type is CONFIDENCE_MASK, the returned vector of images
      contains only one confidence image mask. A segmentation result object that
      contains a list of segmentation masks as images.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If image segmentation failed to run.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    c_result = ImageSegmenterResultC()
    error_msg = ctypes.c_char_p()
    options_c = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    status = self._lib.image_segmenter_segment_for_video_with_options(
        self._handle,
        c_image,
        options_c,
        timestamp_ms,
        ctypes.byref(c_result),
        ctypes.byref(error_msg),
    )
    mediapipe_c_bindings.handle_return_code(
        status, 'Failed to segment for video', error_msg
    )
    result = ImageSegmenterResult.from_ctypes(c_result)
    self._lib.image_segmenter_close_result(ctypes.byref(c_result))
    return result

  def segment_async(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: _ImageProcessingOptions | None = None,
  ) -> None:
    """Sends live image data (an Image with a unique timestamp) to perform image segmentation.

    Only use this method when the ImageSegmenter is created with the live stream
    running mode. The input timestamps should be monotonically increasing for
    adjacent calls of this method. This method will return immediately after the
    input image is accepted. The results will be available via the
    `result_callback` provided in the `ImageSegmenterOptions`. The
    `segment_async` method is designed to process live stream data such as
    camera input. To lower the overall latency, image segmenter may drop the
    input images if needed. In other words, it's not guaranteed to have output
    per input image.

    The `result_callback` prvoides:
      - A segmentation result object that contains a list of segmentation masks
        as images.
      - The input image that the image segmenter runs on.
      - The input timestamp in milliseconds.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input image in milliseconds.
      image_processing_options: Options for image processing.

    Raises:
      ValueError: If the current input timestamp is smaller than what the image
        segmenter has already processed.
    """
    c_image = image._image_ptr  # pylint: disable=protected-access
    error_msg = ctypes.c_char_p()
    options_c = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    status = self._lib.image_segmenter_segment_async_with_options(
        self._handle,
        c_image,
        options_c,
        timestamp_ms,
        ctypes.byref(error_msg),
    )
    mediapipe_c_bindings.handle_return_code(
        status, 'Failed to segment image asynchronously', error_msg
    )

  @property
  def labels(self) -> list[str]:
    """Get the category label list the ImageSegmenter can recognize.

    For CATEGORY_MASK type, the index in the category mask corresponds to the
    category in the label list.
    For CONFIDENCE_MASK type, the output mask list at index corresponds to the
    category in the label list.

    If there is no label map provided in the model file, empty label list is
    returned.
    """
    if not self._labels:
      c_labels = MpStringListC()
      self._lib.image_segmenter_get_labels(
          self._handle,
          ctypes.byref(c_labels),
      )
      self._labels = []
      for i in range(c_labels.num_strings):
        c_label = c_labels.strings[i]
        label = ctypes.string_at(c_label).decode('utf-8')
        self._labels.append(label)
      self._lib.MpStringListFree(ctypes.byref(c_labels))
    return self._labels

  def close(self):
    """Closes ImageSegmenter."""
    if self._handle:
      error_msg = ctypes.c_char_p()
      status = self._lib.image_segmenter_close(
          self._handle, ctypes.byref(error_msg)
      )
      mediapipe_c_bindings.handle_return_code(
          status, 'Failed to close ImageSegmenter', error_msg
      )
    self._handle = None
    self._lib.close()

  def __enter__(self):
    """Returns `self` upon entering the runtime context."""
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Closes ImageSegmenters and exits the context manager.

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
