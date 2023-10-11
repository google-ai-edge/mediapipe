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
"""Writes metadata and creates model asset bundle for face stylizer."""

import os
import tempfile
from typing import List

from mediapipe.tasks.python.metadata.metadata_writers import metadata_writer
from mediapipe.tasks.python.metadata.metadata_writers import model_asset_bundle_utils

_MODEL_NAME = "FaceStylizer"
_MODEL_DESCRIPTION = "Performs face stylization on images."
_FACE_DETECTOR_MODEL = "face_detector.tflite"
_FACE_LANDMARKS_DETECTOR_MODEL = "face_landmarks_detector.tflite"
_FACE_STYLIZER_MODEL = "face_stylizer.tflite"
_FACE_STYLIZER_TASK = "face_stylizer.task"


class MetadataWriter:
  """MetadataWriter to write the metadata for face stylizer."""

  def __init__(
      self,
      face_detector_model_buffer: bytearray,
      face_landmarks_detector_model_buffer: bytearray,
      face_stylizer_metadata_writer: metadata_writer.MetadataWriter,
  ) -> None:
    """Initializes MetadataWriter to write the metadata and model asset bundle.

    Args:
      face_detector_model_buffer: A valid flatbuffer loaded from the face
        detector TFLite model file with metadata already packed inside.
      face_landmarks_detector_model_buffer: A valid flatbuffer loaded from the
        face landmarks detector TFLite model file with metadata already packed
        inside.
      face_stylizer_metadata_writer: Metadata writer to write face stylizer
        metadata into the TFLite file.
    """
    self._face_detector_model_buffer = face_detector_model_buffer
    self._face_landmarks_detector_model_buffer = (
        face_landmarks_detector_model_buffer
    )
    self._face_stylizer_metadata_writer = face_stylizer_metadata_writer
    self._temp_folder = tempfile.TemporaryDirectory()

  def __del__(self):
    if os.path.exists(self._temp_folder.name):
      self._temp_folder.cleanup()

  @classmethod
  def create(
      cls,
      face_stylizer_model_buffer: bytearray,
      face_detector_model_buffer: bytearray,
      face_landmarks_detector_model_buffer: bytearray,
      input_norm_mean: List[float],
      input_norm_std: List[float],
  ) -> "MetadataWriter":
    """Creates MetadataWriter to write the metadata for face stylizer.

    The parameters required in this method are mandatory when using MediaPipe
    Tasks.

    Note that only the output TFLite is used for deployment. The output JSON
    content is used to interpret the metadata content.

    Args:
      face_stylizer_model_buffer: A valid flatbuffer loaded from the face
        stylizer TFLite model file.
      face_detector_model_buffer: A valid flatbuffer loaded from the face
        detector TFLite model file with metadata already packed inside.
      face_landmarks_detector_model_buffer: A valid flatbuffer loaded from the
        face landmarks detector TFLite model file with metadata already packed
        inside.
      input_norm_mean: the mean value used in the input tensor normalization for
        face stylizer model [1].
      input_norm_std: the std value used in the input tensor normalizarion for
        face stylizer model [1].

      [1]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L389

    Returns:
      A MetadataWriter object.
    """
    face_stylizer_writer = metadata_writer.MetadataWriter(
        face_stylizer_model_buffer
    )
    face_stylizer_writer.add_general_info(_MODEL_NAME, _MODEL_DESCRIPTION)
    face_stylizer_writer.add_image_input(input_norm_mean, input_norm_std)
    return cls(
        face_detector_model_buffer,
        face_landmarks_detector_model_buffer,
        face_stylizer_writer,
    )

  def populate(self):
    """Populates the metadata and creates model asset bundle.

    Note that only the output model asset bundle is used for deployment.
    The output JSON content is used to interpret the face stylizer metadata
    content.

    Returns:
      A tuple of (model_asset_bundle_in_bytes, metadata_json_content)
    """
    # Write metadata into the face stylizer TFLite model.
    face_stylizer_model_buffer, face_stylizer_metadata_json = (
        self._face_stylizer_metadata_writer.populate()
    )
    # Create the model asset bundle for the face stylizer task.
    face_stylizer_models = {
        _FACE_DETECTOR_MODEL: self._face_detector_model_buffer,
        _FACE_LANDMARKS_DETECTOR_MODEL: (
            self._face_landmarks_detector_model_buffer
        ),
        _FACE_STYLIZER_MODEL: face_stylizer_model_buffer,
    }
    output_path = os.path.join(self._temp_folder.name, _FACE_STYLIZER_TASK)
    model_asset_bundle_utils.create_model_asset_bundle(
        face_stylizer_models, output_path
    )
    with open(output_path, "rb") as f:
      face_stylizer_model_bundle_buffer = f.read()
    return face_stylizer_model_bundle_buffer, face_stylizer_metadata_json
