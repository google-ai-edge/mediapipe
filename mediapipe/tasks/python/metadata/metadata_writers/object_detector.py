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
# ==============================================================================
"""Writes metadata and label file to the Object Detector models."""

import dataclasses
from typing import List, Optional

import flatbuffers
from mediapipe.tasks.metadata import metadata_schema_py_generated as _metadata_fb
from mediapipe.tasks.metadata import object_detector_metadata_schema_py_generated as _detector_metadata_fb
from mediapipe.tasks.python.metadata import metadata
from mediapipe.tasks.python.metadata.metadata_writers import metadata_info
from mediapipe.tasks.python.metadata.metadata_writers import metadata_writer

_MODEL_NAME = "ObjectDetector"
_MODEL_DESCRIPTION = (
    "Identify which of a known set of objects might be present and provide "
    "information about their positions within the given image or a video "
    "stream."
)

# Metadata Schema file for object detector.
_FLATC_METADATA_SCHEMA_FILE = metadata.get_path_to_datafile(
    "../../../metadata/object_detector_metadata_schema.fbs",
)

# Metadata name in custom metadata field. The metadata name is used to get
# object detector metadata from SubGraphMetadata.custom_metadata and shouldn't
# be changed.
_METADATA_NAME = "DETECTOR_METADATA"


@dataclasses.dataclass
class FixedAnchor:
  """A fixed size anchor."""

  x_center: float
  y_center: float
  width: Optional[float]
  height: Optional[float]


@dataclasses.dataclass
class FixedAnchorsSchema:
  """The schema for a list of anchors with fixed size."""

  anchors: List[FixedAnchor]


@dataclasses.dataclass
class SsdAnchorsOptions:
  """The ssd anchors options used in object detector model."""

  fixed_anchors_schema: Optional[FixedAnchorsSchema]


@dataclasses.dataclass
class TensorsDecodingOptions:
  """The decoding options to convert model output tensors to detections."""

  # The number of output classes predicted by the detection model.
  num_classes: int
  # The number of output boxes predicted by the detection model.
  num_boxes: int
  # The number of output values per boxes predicted by the detection
  # model. The values contain bounding boxes, keypoints, etc.
  num_coords: int
  # The offset of keypoint coordinates in the location tensor.
  keypoint_coord_offset: int
  # The number of predicted keypoints.
  num_keypoints: int
  # The dimension of each keypoint, e.g. number of values predicted for each
  # keypoint.
  num_values_per_keypoint: int
  # Parameters for decoding SSD detection model.
  x_scale: float
  y_scale: float
  w_scale: float
  h_scale: float
  # Whether to apply exponential on box size.
  apply_exponential_on_box_size: bool
  # Whether to apply sigmod function on the score.
  sigmoid_score: bool


# Create an individual method for getting the metadata json file, so that it can
# be used as a standalone util.
def convert_to_json(metadata_buffer: bytearray) -> str:
  """Converts the metadata into a json string.

  Args:
    metadata_buffer: valid metadata buffer in bytes.

  Returns:
    Metadata in JSON format.

  Raises:
    ValueError: error occurred when parsing the metadata schema file.
  """
  return metadata.convert_to_json(
      metadata_buffer,
      custom_metadata_schema={_METADATA_NAME: _FLATC_METADATA_SCHEMA_FILE},
  )


class ObjectDetectorOptionsMd(metadata_info.CustomMetadataMd):
  """Object detector options metadata."""

  _METADATA_FILE_IDENTIFIER = b"V001"

  def __init__(
      self,
      ssd_anchors_options: SsdAnchorsOptions,
      tensors_decoding_options: TensorsDecodingOptions,
  ) -> None:
    """Creates an ObjectDetectorOptionsMd object.

    Args:
      ssd_anchors_options: the ssd anchors options associated to the object
        detector model.
      tensors_decoding_options: the tensors decoding options used to decode the
        object detector model output.
    """
    if ssd_anchors_options.fixed_anchors_schema is None:
      raise ValueError(
          "Currently only support FixedAnchorsSchema, which cannot be found"
          " in ssd_anchors_options."
      )
    self.ssd_anchors_options = ssd_anchors_options
    self.tensors_decoding_options = tensors_decoding_options
    super().__init__(name=_METADATA_NAME)

  def create_metadata(self) -> _metadata_fb.CustomMetadataT:
    """Creates the image segmenter options metadata.

    Returns:
      A Flatbuffers Python object of the custom metadata including object
      detector options metadata.
    """
    detector_options = _detector_metadata_fb.ObjectDetectorOptionsT()

    # Set ssd_anchors_options.
    ssd_anchors_options = _detector_metadata_fb.SsdAnchorsOptionsT()
    fixed_anchors_schema = _detector_metadata_fb.FixedAnchorsSchemaT()
    fixed_anchors_schema.anchors = []
    for anchor in self.ssd_anchors_options.fixed_anchors_schema.anchors:
      anchor_t = _detector_metadata_fb.FixedAnchorT()
      anchor_t.xCenter = anchor.x_center
      anchor_t.yCenter = anchor.y_center
      anchor_t.width = anchor.width
      anchor_t.height = anchor.height
      fixed_anchors_schema.anchors.append(anchor_t)
    ssd_anchors_options.fixedAnchorsSchema = fixed_anchors_schema
    detector_options.ssdAnchorsOptions = ssd_anchors_options

    # Set tensors_decoding_options.
    tensors_decoding_options = _detector_metadata_fb.TensorsDecodingOptionsT()
    tensors_decoding_options.numClasses = (
        self.tensors_decoding_options.num_classes
    )
    tensors_decoding_options.numBoxes = self.tensors_decoding_options.num_boxes
    tensors_decoding_options.numCoords = (
        self.tensors_decoding_options.num_coords
    )
    tensors_decoding_options.keypointCoordOffset = (
        self.tensors_decoding_options.keypoint_coord_offset
    )
    tensors_decoding_options.numKeypoints = (
        self.tensors_decoding_options.num_keypoints
    )
    tensors_decoding_options.numValuesPerKeypoint = (
        self.tensors_decoding_options.num_values_per_keypoint
    )
    tensors_decoding_options.xScale = self.tensors_decoding_options.x_scale
    tensors_decoding_options.yScale = self.tensors_decoding_options.y_scale
    tensors_decoding_options.wScale = self.tensors_decoding_options.w_scale
    tensors_decoding_options.hScale = self.tensors_decoding_options.h_scale
    tensors_decoding_options.applyExponentialOnBoxSize = (
        self.tensors_decoding_options.apply_exponential_on_box_size
    )
    tensors_decoding_options.sigmoidScore = (
        self.tensors_decoding_options.sigmoid_score
    )
    detector_options.tensorsDecodingOptions = tensors_decoding_options

    # Get the object detector options flatbuffer.
    b = flatbuffers.Builder(0)
    b.Finish(detector_options.Pack(b), self._METADATA_FILE_IDENTIFIER)
    detector_options_buf = b.Output()

    # Add the object detector options flatbuffer in custom metadata.
    custom_metadata = _metadata_fb.CustomMetadataT()
    custom_metadata.name = self.name
    custom_metadata.data = detector_options_buf
    return custom_metadata


class MetadataWriter(metadata_writer.MetadataWriterBase):
  """MetadataWriter to write the metadata into the object detector."""

  @classmethod
  def create_for_models_with_nms(
      cls,
      model_buffer: bytearray,
      input_norm_mean: List[float],
      input_norm_std: List[float],
      labels: metadata_writer.Labels,
      score_calibration: Optional[metadata_writer.ScoreCalibration] = None,
  ) -> "MetadataWriter":
    """Creates MetadataWriter to write the metadata for object detector with postprocessing in the model.

    This method create a metadata writer for the models with postprocessing [1].

    The parameters required in this method are mandatory when using MediaPipe
    Tasks.

    Example usage:
      metadata_writer = object_detector.Metadatawriter.create(model_buffer, ...)
      tflite_content, json_content = metadata_writer.populate()

    When calling `populate` function in this class, it returns TfLite content
    and JSON content. Note that only the output TFLite is used for deployment.
    The output JSON content is used to interpret the metadata content.

    Args:
      model_buffer: A valid flatbuffer loaded from the TFLite model file.
      input_norm_mean: the mean value used in the input tensor normalization
        [2].
      input_norm_std: the std value used in the input tensor normalizarion [2].
      labels: an instance of Labels helper class used in the output
        classification tensor [3].
      score_calibration: A container of the score calibration operation [4] in
        the classification tensor. Optional if the model does not use score
        calibration.
      [1]:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/detection_postprocess.cc
      [2]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L389
      [3]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L99
      [4]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L456

    Returns:
      A MetadataWriter object.
    """
    writer = metadata_writer.MetadataWriter(model_buffer)
    writer.add_general_info(_MODEL_NAME, _MODEL_DESCRIPTION)
    writer.add_image_input(input_norm_mean, input_norm_std)
    writer.add_detection_output(labels, score_calibration)
    return cls(writer)

  @classmethod
  def create_for_models_without_nms(
      cls,
      model_buffer: bytearray,
      input_norm_mean: List[float],
      input_norm_std: List[float],
      labels: metadata_writer.Labels,
      ssd_anchors_options: SsdAnchorsOptions,
      tensors_decoding_options: TensorsDecodingOptions,
      output_tensors_order: metadata_info.RawDetectionOutputTensorsOrder = metadata_info.RawDetectionOutputTensorsOrder.UNSPECIFIED,
  ) -> "MetadataWriter":
    """Creates MetadataWriter to write the metadata for object detector without postprocessing in the model.

    This method create a metadata writer for the models without postprocessing
    [1].

    The parameters required in this method are mandatory when using MediaPipe
    Tasks.

    Example usage:
      metadata_writer = object_detector.Metadatawriter.create(model_buffer, ...)
      tflite_content, json_content = metadata_writer.populate()

    When calling `populate` function in this class, it returns TfLite content
    and JSON content. Note that only the output TFLite is used for deployment.
    The output JSON content is used to interpret the metadata content.

    Args:
      model_buffer: A valid flatbuffer loaded from the TFLite model file.
      input_norm_mean: the mean value used in the input tensor normalization
        [2].
      input_norm_std: the std value used in the input tensor normalizarion [2].
      labels: an instance of Labels helper class used in the output
        classification tensor [3].
      ssd_anchors_options: the ssd anchors options associated to the object
        detector model.
      tensors_decoding_options: the tensors decoding options used to decode the
        object detector model output.
      output_tensors_order: the order of the output tensors.
      [1]:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/detection_postprocess.cc
      [2]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L389
      [3]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L99

    Returns:
      A MetadataWriter object.
    """
    writer = metadata_writer.MetadataWriter(model_buffer)
    writer.add_general_info(_MODEL_NAME, _MODEL_DESCRIPTION)
    writer.add_image_input(input_norm_mean, input_norm_std)
    writer.add_raw_detection_output(
        labels, output_tensors_order=output_tensors_order
    )
    option_md = ObjectDetectorOptionsMd(
        ssd_anchors_options, tensors_decoding_options
    )
    writer.add_custom_metadata(option_md)
    return cls(writer)

  def populate(self) -> "tuple[bytearray, str]":
    model_buf, _ = super().populate()
    metadata_buf = metadata.get_metadata_buffer(model_buf)
    json_content = convert_to_json(metadata_buf)
    return model_buf, json_content
