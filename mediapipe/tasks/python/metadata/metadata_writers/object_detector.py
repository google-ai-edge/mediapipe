# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
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

from typing import List, Optional

from mediapipe.tasks.python.metadata.metadata_writers import metadata_writer

_MODEL_NAME = "ObjectDetector"
_MODEL_DESCRIPTION = (
    "Identify which of a known set of objects might be present and provide "
    "information about their positions within the given image or a video "
    "stream."
)


class MetadataWriter(metadata_writer.MetadataWriterBase):
  """MetadataWriter to write the metadata into the object detector."""

  @classmethod
  def create(
      cls,
      model_buffer: bytearray,
      input_norm_mean: List[float],
      input_norm_std: List[float],
      labels: metadata_writer.Labels,
      score_calibration: Optional[metadata_writer.ScoreCalibration] = None,
  ) -> "MetadataWriter":
    """Creates MetadataWriter to write the metadata for image classifier.

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
        [1].
      input_norm_std: the std value used in the input tensor normalizarion [1].
      labels: an instance of Labels helper class used in the output
        classification tensor [2].
      score_calibration: A container of the score calibration operation [3] in
        the classification tensor. Optional if the model does not use score
        calibration.
      [1]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L389
      [2]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L99
      [3]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L456

    Returns:
      A MetadataWriter object.
    """
    writer = metadata_writer.MetadataWriter(model_buffer)
    writer.add_general_info(_MODEL_NAME, _MODEL_DESCRIPTION)
    writer.add_image_input(input_norm_mean, input_norm_std)
    writer.add_detection_output(labels, score_calibration)
    return cls(writer)
