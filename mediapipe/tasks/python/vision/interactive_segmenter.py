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

import dataclasses
import enum
from typing import List, Optional

from mediapipe.python import packet_creator
from mediapipe.python import packet_getter
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.tasks.cc.vision.image_segmenter.proto import image_segmenter_graph_options_pb2
from mediapipe.tasks.cc.vision.image_segmenter.proto import segmenter_options_pb2
from mediapipe.tasks.python.components.containers import keypoint as keypoint_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import task_info as task_info_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import base_vision_task_api
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode
from mediapipe.util import render_data_pb2

_BaseOptions = base_options_module.BaseOptions
_RenderDataProto = render_data_pb2.RenderData
_SegmenterOptionsProto = segmenter_options_pb2.SegmenterOptions
_ImageSegmenterGraphOptionsProto = (
    image_segmenter_graph_options_pb2.ImageSegmenterGraphOptions
)
_RunningMode = vision_task_running_mode.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions
_TaskInfo = task_info_module.TaskInfo

_CONFIDENCE_MASKS_STREAM_NAME = 'confidence_masks'
_CONFIDENCE_MASKS_TAG = 'CONFIDENCE_MASKS'
_CATEGORY_MASK_STREAM_NAME = 'category_mask'
_CATEGORY_MASK_TAG = 'CATEGORY_MASK'
_IMAGE_IN_STREAM_NAME = 'image_in'
_IMAGE_OUT_STREAM_NAME = 'image_out'
_ROI_STREAM_NAME = 'roi_in'
_ROI_TAG = 'ROI'
_NORM_RECT_STREAM_NAME = 'norm_rect_in'
_NORM_RECT_TAG = 'NORM_RECT'
_IMAGE_TAG = 'IMAGE'
_TASK_GRAPH_NAME = (
    'mediapipe.tasks.vision.interactive_segmenter.InteractiveSegmenterGraph'
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


@dataclasses.dataclass
class InteractiveSegmenterOptions:
  """Options for the interactive segmenter task.

  Attributes:
    base_options: Base options for the interactive segmenter task.
    output_confidence_masks: Whether to output confidence masks.
    output_category_mask: Whether to output category mask.
  """

  base_options: _BaseOptions
  output_confidence_masks: bool = True
  output_category_mask: bool = False

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _ImageSegmenterGraphOptionsProto:
    """Generates an ImageSegmenterGraphOptions protobuf object."""
    base_options_proto = self.base_options.to_pb2()
    base_options_proto.use_stream_mode = False
    segmenter_options_proto = _SegmenterOptionsProto()
    return _ImageSegmenterGraphOptionsProto(
        base_options=base_options_proto,
        segmenter_options=segmenter_options_proto,
    )


@dataclasses.dataclass
class RegionOfInterest:
  """The Region-Of-Interest (ROI) to interact with."""

  class Format(enum.Enum):
    UNSPECIFIED = 0
    KEYPOINT = 1

  format: Format
  keypoint: Optional[keypoint_module.NormalizedKeypoint] = None


def _convert_roi_to_render_data(roi: RegionOfInterest) -> _RenderDataProto:
  """Converts region of interest to render data proto."""
  result = _RenderDataProto()

  if roi is not None:
    if roi.format == RegionOfInterest.Format.UNSPECIFIED:
      raise ValueError('RegionOfInterest format not specified.')

    elif roi.format == RegionOfInterest.Format.KEYPOINT:
      if roi.keypoint is not None:
        annotation = result.render_annotations.add()
        annotation.color.r = 255
        point = annotation.point
        point.normalized = True
        point.x = roi.keypoint.x
        point.y = roi.keypoint.y
        return result
  else:
    raise ValueError('Please specify the Region-of-interest for segmentation.')

  raise ValueError('Unrecognized format.')


class InteractiveSegmenter(base_vision_task_api.BaseVisionTaskApi):
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

    output_streams = [
        ':'.join([_IMAGE_TAG, _IMAGE_OUT_STREAM_NAME]),
    ]

    if options.output_confidence_masks:
      output_streams.append(
          ':'.join([_CONFIDENCE_MASKS_TAG, _CONFIDENCE_MASKS_STREAM_NAME])
      )

    if options.output_category_mask:
      output_streams.append(
          ':'.join([_CATEGORY_MASK_TAG, _CATEGORY_MASK_STREAM_NAME])
      )

    task_info = _TaskInfo(
        task_graph=_TASK_GRAPH_NAME,
        input_streams=[
            ':'.join([_IMAGE_TAG, _IMAGE_IN_STREAM_NAME]),
            ':'.join([_ROI_TAG, _ROI_STREAM_NAME]),
            ':'.join([_NORM_RECT_TAG, _NORM_RECT_STREAM_NAME]),
        ],
        output_streams=output_streams,
        task_options=options,
    )
    return cls(
        task_info.generate_graph_config(enable_flow_limiting=False),
        _RunningMode.IMAGE,
        None,
    )

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
    normalized_rect = self.convert_to_normalized_rect(
        image_processing_options, image, roi_allowed=False
    )
    render_data_proto = _convert_roi_to_render_data(roi)
    output_packets = self._process_image_data({
        _IMAGE_IN_STREAM_NAME: packet_creator.create_image(image),
        _ROI_STREAM_NAME: packet_creator.create_proto(render_data_proto),
        _NORM_RECT_STREAM_NAME: packet_creator.create_proto(
            normalized_rect.to_pb2()
        ),
    })
    segmentation_result = InteractiveSegmenterResult()

    if _CONFIDENCE_MASKS_STREAM_NAME in output_packets:
      segmentation_result.confidence_masks = packet_getter.get_image_list(
          output_packets[_CONFIDENCE_MASKS_STREAM_NAME]
      )

    if _CATEGORY_MASK_STREAM_NAME in output_packets:
      segmentation_result.category_mask = packet_getter.get_image(
          output_packets[_CATEGORY_MASK_STREAM_NAME]
      )

    return segmentation_result
