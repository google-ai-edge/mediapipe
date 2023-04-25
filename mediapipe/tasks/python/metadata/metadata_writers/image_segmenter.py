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
# ==============================================================================
"""Writes metadata and label file to the image segmenter models."""
import enum
from typing import List, Optional

import flatbuffers
from mediapipe.tasks.metadata import image_segmenter_metadata_schema_py_generated as _segmenter_metadata_fb
from mediapipe.tasks.metadata import metadata_schema_py_generated as _metadata_fb
from mediapipe.tasks.python.metadata import metadata
from mediapipe.tasks.python.metadata.metadata_writers import metadata_info
from mediapipe.tasks.python.metadata.metadata_writers import metadata_writer


_MODEL_NAME = "ImageSegmenter"
_MODEL_DESCRIPTION = (
    "Semantic image segmentation predicts whether each pixel "
    "of an image is associated with a certain class."
)

# Metadata Schema file for image segmenter.
_FLATC_METADATA_SCHEMA_FILE = metadata.get_path_to_datafile(
    "../../../metadata/image_segmenter_metadata_schema.fbs",
)

# Metadata name in custom metadata field. The metadata name is used to get
# image segmenter metadata from SubGraphMetadata.custom_metadata and
# shouldn't be changed.
_METADATA_NAME = "SEGMENTER_METADATA"


class Activation(enum.Enum):
  NONE = 0
  SIGMOID = 1
  SOFTMAX = 2


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


class ImageSegmenterOptionsMd(metadata_info.CustomMetadataMd):
  """Image segmenter options metadata."""

  _METADATA_FILE_IDENTIFIER = b"V001"

  def __init__(self, activation: Activation) -> None:
    """Creates an ImageSegmenterOptionsMd object.

    Args:
      activation: activation function of the output layer in the image
        segmenter.
    """
    self.activation = activation
    super().__init__(name=_METADATA_NAME)

  def create_metadata(self) -> _metadata_fb.CustomMetadataT:
    """Creates the image segmenter options metadata.

    Returns:
      A Flatbuffers Python object of the custom metadata including image
      segmenter options metadata.
    """
    segmenter_options = _segmenter_metadata_fb.ImageSegmenterOptionsT()
    segmenter_options.activation = self.activation.value

    # Get the image segmenter options flatbuffer.
    b = flatbuffers.Builder(0)
    b.Finish(segmenter_options.Pack(b), self._METADATA_FILE_IDENTIFIER)
    segmenter_options_buf = b.Output()

    # Add the image segmenter options flatbuffer in custom metadata.
    custom_metadata = _metadata_fb.CustomMetadataT()
    custom_metadata.name = self.name
    custom_metadata.data = segmenter_options_buf
    return custom_metadata


class MetadataWriter(metadata_writer.MetadataWriterBase):
  """MetadataWriter to write the metadata for image segmenter."""

  @classmethod
  def create(
      cls,
      model_buffer: bytearray,
      input_norm_mean: List[float],
      input_norm_std: List[float],
      labels: Optional[metadata_writer.Labels] = None,
      activation: Optional[Activation] = None,
  ) -> "MetadataWriter":
    """Creates MetadataWriter to write the metadata for image segmenter.

    The parameters required in this method are mandatory when using MediaPipe
    Tasks.

    Example usage:
      metadata_writer = image_segmenter.Metadatawriter.create(model_buffer, ...)
      tflite_content, json_content = metadata_writer.populate()

    When calling `populate` function in this class, it returns TfLite content
    and JSON content. Note that only the output TFLite is used for deployment.
    The output JSON content is used to interpret the metadata content.

    Args:
      model_buffer: A valid flatbuffer loaded from the TFLite model file.
      input_norm_mean: the mean value used in the input tensor normalization
        [1].
      input_norm_std: the std value used in the input tensor normalizarion [1].
      labels: an instance of Labels helper class used in the output category
        tensor [2].
      activation: activation function for the output layer.
      [1]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L389
      [2]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L116

    Returns:
      A MetadataWriter object.
    """
    writer = metadata_writer.MetadataWriter(model_buffer)
    writer.add_general_info(_MODEL_NAME, _MODEL_DESCRIPTION)
    writer.add_image_input(input_norm_mean, input_norm_std)
    writer.add_segmentation_output(labels=labels)
    if activation is not None:
      option_md = ImageSegmenterOptionsMd(activation)
      writer.add_custom_metadata(option_md)
    return cls(writer)

  def populate(self) -> tuple[bytearray, str]:
    model_buf, _ = super().populate()
    metadata_buf = metadata.get_metadata_buffer(model_buf)
    json_content = convert_to_json(metadata_buf)
    return model_buf, json_content
