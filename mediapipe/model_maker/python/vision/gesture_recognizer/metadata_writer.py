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
"""Writes metadata and creates model asset bundle for gesture recognizer."""

import dataclasses
import os
import tempfile
from typing import Union

import tensorflow as tf
from mediapipe.tasks.python.metadata.metadata_writers import metadata_writer
from mediapipe.tasks.python.metadata.metadata_writers import model_asset_bundle_utils

_HAND_DETECTOR_TFLITE_NAME = "hand_detector.tflite"
_HAND_LANDMARKS_DETECTOR_TFLITE_NAME = "hand_landmarks_detector.tflite"
_HAND_LANDMARKER_BUNDLE_NAME = "hand_landmarker.task"
_HAND_GESTURE_RECOGNIZER_BUNDLE_NAME = "hand_gesture_recognizer.task"
_GESTURE_EMBEDDER_TFLITE_NAME = "gesture_embedder.tflite"
_CANNED_GESTURE_CLASSIFIER_TFLITE_NAME = "canned_gesture_classifier.tflite"
_CUSTOM_GESTURE_CLASSIFIER_TFLITE_NAME = "custom_gesture_classifier.tflite"

_MODEL_NAME = "HandGestureRecognition"
_MODEL_DESCRIPTION = "Recognize the hand gesture in the image."

_INPUT_NAME = "embedding"
_INPUT_DESCRIPTION = "Embedding feature vector from gesture embedder."
_OUTPUT_NAME = "scores"
_OUTPUT_DESCRIPTION = "Hand gesture category scores."


@dataclasses.dataclass
class GestureClassifierOptions:
  """Options to write metadata for gesture classifier.

  Attributes:
    model_buffer: Gesture classifier TFLite model buffer.
    labels: Labels for the gesture classifier.
    score_thresholding: Parameters to performs thresholding on output tensor
      values [1].
    [1]:
      https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L468
  """
  model_buffer: bytearray
  labels: metadata_writer.Labels
  score_thresholding: metadata_writer.ScoreThresholding


def read_file(file_path: str, mode: str = "rb") -> Union[str, bytes]:
  with tf.io.gfile.GFile(file_path, mode) as f:
    return f.read()


class HandLandmarkerMetadataWriter:
  """MetadataWriter to write the model asset bundle for HandLandmarker."""

  def __init__(
      self,
      hand_detector_model_buffer: bytearray,
      hand_landmarks_detector_model_buffer: bytearray,
  ) -> None:
    """Initializes HandLandmarkerMetadataWriter to write model asset bundle.

    Args:
      hand_detector_model_buffer: A valid flatbuffer *with* metadata loaded from
        the TFLite hand detector model file.
      hand_landmarks_detector_model_buffer: A valid flatbuffer *with* metadata
        loaded from the TFLite hand landmarks detector model file.
    """
    self._hand_detector_model_buffer = hand_detector_model_buffer
    self._hand_landmarks_detector_model_buffer = hand_landmarks_detector_model_buffer
    self._temp_folder = tempfile.TemporaryDirectory()

  def __del__(self):
    if os.path.exists(self._temp_folder.name):
      self._temp_folder.cleanup()

  def populate(self):
    """Creates the model asset bundle for hand landmarker task.

    Returns:
      Model asset bundle in bytes
    """
    landmark_models = {
        _HAND_DETECTOR_TFLITE_NAME:
            self._hand_detector_model_buffer,
        _HAND_LANDMARKS_DETECTOR_TFLITE_NAME:
            self._hand_landmarks_detector_model_buffer
    }
    output_hand_landmarker_path = os.path.join(self._temp_folder.name,
                                               _HAND_LANDMARKER_BUNDLE_NAME)
    model_asset_bundle_utils.create_model_asset_bundle(
        landmark_models, output_hand_landmarker_path
    )
    hand_landmarker_model_buffer = read_file(output_hand_landmarker_path)
    return hand_landmarker_model_buffer


class MetadataWriter:
  """MetadataWriter to write the metadata and the model asset bundle."""

  def __init__(
      self, hand_detector_model_buffer: bytearray,
      hand_landmarks_detector_model_buffer: bytearray,
      gesture_embedder_model_buffer: bytearray,
      canned_gesture_classifier_model_buffer: bytearray,
      custom_gesture_classifier_metadata_writer: metadata_writer.MetadataWriter
  ) -> None:
    """Initialize MetadataWriter to write the metadata and model asset bundle.

    Args:
      hand_detector_model_buffer: A valid flatbuffer *with* metadata loaded from
        the TFLite hand detector model file.
      hand_landmarks_detector_model_buffer: A valid flatbuffer *with* metadata
        loaded from the TFLite hand landmarks detector model file.
      gesture_embedder_model_buffer: A valid flatbuffer *with* metadata loaded
        from the TFLite gesture embedder model file.
      canned_gesture_classifier_model_buffer: A valid flatbuffer *with* metadata
        loaded from the TFLite canned gesture classifier model file.
      custom_gesture_classifier_metadata_writer: Metadata writer to write custom
        gesture classifier metadata into the TFLite file.
    """
    self._hand_landmarker_metadata_writer = HandLandmarkerMetadataWriter(
        hand_detector_model_buffer, hand_landmarks_detector_model_buffer)
    self._gesture_embedder_model_buffer = gesture_embedder_model_buffer
    self._canned_gesture_classifier_model_buffer = canned_gesture_classifier_model_buffer
    self._custom_gesture_classifier_metadata_writer = custom_gesture_classifier_metadata_writer
    self._temp_folder = tempfile.TemporaryDirectory()

  def __del__(self):
    if os.path.exists(self._temp_folder.name):
      self._temp_folder.cleanup()

  @classmethod
  def create(
      cls,
      hand_detector_model_buffer: bytearray,
      hand_landmarks_detector_model_buffer: bytearray,
      gesture_embedder_model_buffer: bytearray,
      canned_gesture_classifier_model_buffer: bytearray,
      custom_gesture_classifier_options: GestureClassifierOptions,
  ) -> "MetadataWriter":
    """Creates MetadataWriter to write the metadata for gesture recognizer.

    Args:
      hand_detector_model_buffer: A valid flatbuffer *with* metadata loaded from
        the TFLite hand detector model file.
      hand_landmarks_detector_model_buffer: A valid flatbuffer *with* metadata
        loaded from the TFLite hand landmarks detector model file.
      gesture_embedder_model_buffer: A valid flatbuffer *with* metadata loaded
        from the TFLite gesture embedder model file.
      canned_gesture_classifier_model_buffer: A valid flatbuffer *with* metadata
        loaded from the TFLite canned gesture classifier model file.
      custom_gesture_classifier_options: Custom gesture classifier options to
        write custom gesture classifier metadata into the TFLite file.

    Returns:
      An MetadataWrite object.
    """
    writer = metadata_writer.MetadataWriter.create(
        custom_gesture_classifier_options.model_buffer)
    writer.add_general_info(_MODEL_NAME, _MODEL_DESCRIPTION)
    writer.add_feature_input(name=_INPUT_NAME, description=_INPUT_DESCRIPTION)
    writer.add_classification_output(
        labels=custom_gesture_classifier_options.labels,
        score_thresholding=custom_gesture_classifier_options.score_thresholding,
        name=_OUTPUT_NAME,
        description=_OUTPUT_DESCRIPTION)
    return cls(hand_detector_model_buffer, hand_landmarks_detector_model_buffer,
               gesture_embedder_model_buffer,
               canned_gesture_classifier_model_buffer, writer)

  def populate(self):
    """Populates the metadata and creates model asset bundle.

    Note that only the output model asset bundle is used for deployment.
    The output JSON content is used to interpret the custom gesture classifier
    metadata content.

    Returns:
      A tuple of (model_asset_bundle_in_bytes, metadata_json_content)
    """
    # Creates the model asset bundle for hand landmarker task.
    hand_landmarker_model_buffer = self._hand_landmarker_metadata_writer.populate(
    )

    # Write metadata into custom gesture classifier model.
    self._custom_gesture_classifier_model_buffer, custom_gesture_classifier_metadata_json = self._custom_gesture_classifier_metadata_writer.populate(
    )
    # Creates the model asset bundle for hand gesture recognizer sub graph.
    hand_gesture_recognizer_models = {
        _GESTURE_EMBEDDER_TFLITE_NAME:
            self._gesture_embedder_model_buffer,
        _CANNED_GESTURE_CLASSIFIER_TFLITE_NAME:
            self._canned_gesture_classifier_model_buffer,
        _CUSTOM_GESTURE_CLASSIFIER_TFLITE_NAME:
            self._custom_gesture_classifier_model_buffer
    }
    output_hand_gesture_recognizer_path = os.path.join(
        self._temp_folder.name, _HAND_GESTURE_RECOGNIZER_BUNDLE_NAME)
    model_asset_bundle_utils.create_model_asset_bundle(
        hand_gesture_recognizer_models, output_hand_gesture_recognizer_path
    )

    # Creates the model asset bundle for end-to-end hand gesture recognizer
    # graph.
    gesture_recognizer_models = {
        _HAND_LANDMARKER_BUNDLE_NAME:
            hand_landmarker_model_buffer,
        _HAND_GESTURE_RECOGNIZER_BUNDLE_NAME:
            read_file(output_hand_gesture_recognizer_path),
    }

    output_file_path = os.path.join(self._temp_folder.name,
                                    "gesture_recognizer.task")
    model_asset_bundle_utils.create_model_asset_bundle(
        gesture_recognizer_models, output_file_path
    )
    with open(output_file_path, "rb") as f:
      gesture_recognizer_model_buffer = f.read()
    return gesture_recognizer_model_buffer, custom_gesture_classifier_metadata_json
