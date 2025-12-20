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
"""MediaPipe interactive segmenter task."""

import ctypes
import dataclasses
import enum
from typing import List, Optional

from mediapipe.tasks.python.components.containers import keypoint as keypoint_module
from mediapipe.tasks.python.components.containers import keypoint_c as keypoint_c_module
from mediapipe.tasks.python.core import async_result_dispatcher
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import base_options_c as base_options_c_module
from mediapipe.tasks.python.core import mediapipe_c_bindings
from mediapipe.tasks.python.core import mediapipe_c_utils
from mediapipe.tasks.python.core import serial_dispatcher
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision import image_segmenter
from mediapipe.tasks.python.vision.core import image as image_module
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import image_processing_options_c as image_processing_options_c_module

_BaseOptions = base_options_module.BaseOptions
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions
_CFunction = mediapipe_c_utils.CFunction
_AsyncResultDispatcher = async_result_dispatcher.AsyncResultDispatcher


class RegionOfInterestC(ctypes.Structure):
  """The Region-Of-Interest (ROI) to interact with."""

  class Format(enum.IntEnum):
    UNSPECIFIED = 0
    KEYPOINT = 1
    SCRIBBLE = 2

  _fields_ = [
      ('format', ctypes.c_int),
      ('keypoint', ctypes.POINTER(keypoint_c_module.NormalizedKeypointC)),
      ('scribble', ctypes.POINTER(keypoint_c_module.NormalizedKeypointC)),
      ('scribble_count', ctypes.c_uint32),
  ]


class InteractiveSegmenterOptionsC(ctypes.Structure):
  """The MediaPipe Tasks InteractiveSegmenterOptions CTypes struct."""

  _fields_ = [
      ('base_options', base_options_c_module.BaseOptionsC),
      ('output_confidence_masks', ctypes.c_bool),
      ('output_category_mask', ctypes.c_bool),
  ]


_CTYPES_SIGNATURES = (
    mediapipe_c_utils.CStatusFunction(
        'MpInteractiveSegmenterCreate',
        (
            ctypes.POINTER(InteractiveSegmenterOptionsC),
            ctypes.POINTER(ctypes.c_void_p),
        ),
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpInteractiveSegmenterSegmentImage',
        (
            ctypes.c_void_p,
            ctypes.c_void_p,  # image
            ctypes.POINTER(RegionOfInterestC),
            ctypes.POINTER(
                image_processing_options_c_module.ImageProcessingOptionsC
            ),
            ctypes.POINTER(image_segmenter.ImageSegmenterResultC),
        ),
    ),
    mediapipe_c_utils.CFunction(
        'MpInteractiveSegmenterCloseResult',
        [ctypes.POINTER(image_segmenter.ImageSegmenterResultC)],
        None,
    ),
    mediapipe_c_utils.CStatusFunction(
        'MpInteractiveSegmenterClose',
        (ctypes.c_void_p,),
    ),
)


@dataclasses.dataclass
class InteractiveSegmenterResult:
  """Output result of InteractiveSegmenter.

  confidence_masks: multiple masks of float image where, for each mask, each
  pixel represents the prediction confidence, usually in the [0, 1] range.

  category_mask: a category mask of uint8 image where each pixel represents the
  class which the pixel in the original image was predicted to belong to.
  """

  confidence_masks: Optional[List[image_module.Image]] = None
  category_mask: Optional[image_module.Image] = None

  @classmethod
  @doc_controls.do_not_generate_docs
  def from_ctypes(
      cls, c_result: image_segmenter.ImageSegmenterResultC
  ) -> 'InteractiveSegmenterResult':
    """Converts a C ImageSegmenterResult to a Python InteractiveSegmenterResult."""
    base_result = image_segmenter.ImageSegmenterResult.from_ctypes(c_result)
    return cls(
        confidence_masks=base_result.confidence_masks,
        category_mask=base_result.category_mask,
    )


@dataclasses.dataclass
class InteractiveSegmenterOptions:
  """Options for the interactive segmenter task.

  Attributes:
    base_options: Base options for the interactive segmenter task.
    running_mode: The running mode of the task.
    output_confidence_masks: Whether to output confidence masks.
    output_category_mask: Whether to output category mask.
    result_callback: The callback function that is invoked synchronously when
      the current thread is idle.
  """

  base_options: _BaseOptions
  output_confidence_masks: bool = True
  output_category_mask: bool = False

  @doc_controls.do_not_generate_docs
  def to_ctypes(self) -> InteractiveSegmenterOptionsC:
    """Generates an InteractiveSegmenterOptionsC ctypes struct."""
    return InteractiveSegmenterOptionsC(
        base_options=self.base_options.to_ctypes(),
        output_confidence_masks=self.output_confidence_masks,
        output_category_mask=self.output_category_mask,
    )


@dataclasses.dataclass
class RegionOfInterest:
  """The Region-Of-Interest (ROI) to interact with."""

  class Format(enum.Enum):
    UNSPECIFIED = 0
    KEYPOINT = 1

  format: Format
  keypoint: Optional[keypoint_module.NormalizedKeypoint] = None

  @doc_controls.do_not_generate_docs
  def to_ctypes(self) -> RegionOfInterestC:
    """Converts a Python RegionOfInterest to a C RegionOfInterestC."""
    if self.keypoint is not None:
      if self.format == RegionOfInterest.Format.UNSPECIFIED:
        raise ValueError('RegionOfInterest format not specified.')
      elif self.format == RegionOfInterest.Format.KEYPOINT:
        c_roi = RegionOfInterestC(format=self.format.value)
        c_keypoint = keypoint_c_module.NormalizedKeypointC(
            x=self.keypoint.x, y=self.keypoint.y
        )
        c_roi.keypoint = ctypes.pointer(c_keypoint)
        return c_roi
      else:
        raise ValueError(
            'Please specify the Region-of-interest for segmentation.'
        )

    raise ValueError('Unrecognized format.')


class InteractiveSegmenter:
  """Class that performs interactive segmentation on images.

  Users can represent user interaction through `RegionOfInterest`, which gives
  a hint to InteractiveSegmenter to perform segmentation focusing on the given
  region of interest.

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
    - if `output_type` is CATEGORY_MASK, uint8 Image, Image vector of size 1.
    - if `output_type` is CONFIDENCE_MASK, float32 Image list of size
      `channels`.
    - batch is always 1

  An example of such model can be found at:
  https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/metadata/2
  """

  _lib: serial_dispatcher.SerialDispatcher
  _handle: ctypes.c_void_p
  _dispatcher: async_result_dispatcher.AsyncResultDispatcher

  def __init__(
      self,
      lib: serial_dispatcher.SerialDispatcher,
      handle: ctypes.c_void_p,
      dispatcher: async_result_dispatcher.AsyncResultDispatcher,
  ):
    """Initializes the `InteractiveSegmenter` object."""
    self._lib = lib
    self._handle = handle
    self._dispatcher = dispatcher

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'InteractiveSegmenter':
    """Creates an `InteractiveSegmenter` object from a TensorFlow Lite model and the default `InteractiveSegmenterOptions`.

    Note that the created `InteractiveSegmenter` instance is in image mode, for
    performing image segmentation on single image inputs.

    Args:
      model_path: Path to the model.

    Returns:
      `InteractiveSegmenter` object that's created from the model file and the
      default `InteractiveSegmenterOptions`.

    Raises:
      ValueError: If failed to create `InteractiveSegmenter` object from the
        provided file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = InteractiveSegmenterOptions(base_options=base_options)
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(
      cls, options: InteractiveSegmenterOptions
  ) -> 'InteractiveSegmenter':
    """Creates the `InteractiveSegmenter` object from interactive segmenter options.

    Args:
      options: Options for the interactive segmenter task.

    Returns:
      `InteractiveSegmenter` object that's created from `options`.

    Raises:
      ValueError: If failed to create `InteractiveSegmenter` object from
        `InteractiveSegmenterOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """
    lib = mediapipe_c_bindings.load_shared_library(_CTYPES_SIGNATURES)
    dispatcher = _AsyncResultDispatcher(converter=lambda x: x)
    ctypes_options = options.to_ctypes()

    segmenter_handle = ctypes.c_void_p()
    lib.MpInteractiveSegmenterCreate(
        ctypes.byref(ctypes_options), ctypes.byref(segmenter_handle)
    )
    return cls(lib, segmenter_handle, dispatcher)

  def segment(
      self,
      image: image_module.Image,
      roi: RegionOfInterest,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> InteractiveSegmenterResult:
    """Performs the actual segmentation task on the provided MediaPipe Image.

    The image can be of any size with format RGB.

    Args:
      image: MediaPipe Image.
      roi: Optional user-specified region of interest for segmentation.
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
    c_roi = roi.to_ctypes()
    c_result = image_segmenter.ImageSegmenterResultC()
    options_c = (
        ctypes.byref(image_processing_options.to_ctypes())
        if image_processing_options
        else None
    )
    self._lib.MpInteractiveSegmenterSegmentImage(
        self._handle,
        c_image,
        ctypes.byref(c_roi),
        options_c,
        ctypes.byref(c_result),
    )

    py_result = InteractiveSegmenterResult.from_ctypes(c_result)
    self._lib.MpInteractiveSegmenterCloseResult(ctypes.byref(c_result))
    return py_result

  def close(self):
    """Closes the InteractiveSegmenter."""
    if not self._handle:
      return
    self._lib.MpInteractiveSegmenterClose(self._handle)
    self._handle = None
    self._dispatcher.close()
    self._lib.close()

  def __enter__(self):
    """Returns `self` upon entering the runtime context."""
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Shuts down the MediaPipe task instance on exit of the context manager."""
    del exc_type, exc_value, traceback
    self.close()

  def __del__(self):
    self.close()
