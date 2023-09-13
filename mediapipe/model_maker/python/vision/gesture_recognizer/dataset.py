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
"""Gesture recognition dataset library."""

import dataclasses
import os
import random
from typing import List, Optional

import tensorflow as tf

from mediapipe.model_maker.python.core.data import classification_dataset
from mediapipe.model_maker.python.core.utils import model_util
from mediapipe.model_maker.python.vision.gesture_recognizer import constants
from mediapipe.model_maker.python.vision.gesture_recognizer import metadata_writer
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision import hand_landmarker as hand_landmarker_module

_Image = image_module.Image
_HandLandmarker = hand_landmarker_module.HandLandmarker
_HandLandmarkerOptions = hand_landmarker_module.HandLandmarkerOptions
_HandLandmarkerResult = hand_landmarker_module.HandLandmarkerResult


@dataclasses.dataclass
class HandDataPreprocessingParams:
  """A dataclass wraps the hand data preprocessing hyperparameters.

  Attributes:
    shuffle: A boolean controlling if shuffle the dataset. Default to true.
    min_detection_confidence: confidence threshold for hand detection.
  """
  shuffle: bool = True
  min_detection_confidence: float = 0.7


@dataclasses.dataclass
class HandData:
  """A dataclass represents hand data for training gesture recognizer model.

  See https://google.github.io/mediapipe/solutions/hands#mediapipe-hands for
  more details of the hand gesture data API.

  Attributes:
    hand: normalized hand landmarks of shape 21x3 from the screen based
      hand-landmark model.
    world_hand: hand landmarks of shape 21x3 in world coordinates.
    handedness: Collection of handedness confidence of the detected hands (i.e.
      is it a left or right hand).
  """
  hand: List[List[float]]
  world_hand: List[List[float]]
  handedness: List[float]


def _validate_data_sample(data: _HandLandmarkerResult) -> bool:
  """Validates the input hand data sample.

  Args:
    data: input hand data sample.

  Returns:
    False if the input data namedtuple does not contain the fields including
    'multi_hand_landmarks' or 'multi_hand_world_landmarks' or 'multi_handedness'
    or any of these attributes' values are none. Otherwise, True.
  """
  if data.hand_landmarks is None or not data.hand_landmarks:
    return False
  if data.hand_world_landmarks is None or not data.hand_world_landmarks:
    return False
  if data.handedness is None or not data.handedness:
    return False
  return True


def _get_hand_data(all_image_paths: List[str],
                   min_detection_confidence: float) -> List[Optional[HandData]]:
  """Computes hand data (landmarks and handedness) in the input image.

  Args:
    all_image_paths: all input image paths.
    min_detection_confidence: hand detection confidence threshold

  Returns:
    A HandData object. Returns None if no hand is detected.
  """
  hand_data_result = []
  hand_detector_model_buffer = model_util.load_tflite_model_buffer(
      constants.HAND_DETECTOR_TFLITE_MODEL_FILE.get_path()
  )
  hand_landmarks_detector_model_buffer = model_util.load_tflite_model_buffer(
      constants.HAND_LANDMARKS_DETECTOR_TFLITE_MODEL_FILE.get_path()
  )
  hand_landmarker_writer = metadata_writer.HandLandmarkerMetadataWriter(
      hand_detector_model_buffer, hand_landmarks_detector_model_buffer)
  hand_landmarker_options = _HandLandmarkerOptions(
      base_options=base_options_module.BaseOptions(
          model_asset_buffer=hand_landmarker_writer.populate()),
      num_hands=1,
      min_hand_detection_confidence=min_detection_confidence,
      min_hand_presence_confidence=0.5,
      min_tracking_confidence=1,
  )
  with _HandLandmarker.create_from_options(
      hand_landmarker_options) as hand_landmarker:
    for path in all_image_paths:
      tf.compat.v1.logging.info('Loading image %s', path)
      image = _Image.create_from_file(path)
      data = hand_landmarker.detect(image)
      if not _validate_data_sample(data):
        hand_data_result.append(None)
        continue
      hand_landmarks = [[hand_landmark.x, hand_landmark.y, hand_landmark.z]
                        for hand_landmark in data.hand_landmarks[0]]
      hand_world_landmarks = [[
          hand_landmark.x, hand_landmark.y, hand_landmark.z
      ] for hand_landmark in data.hand_world_landmarks[0]]
      handedness_scores = [
          handedness.score for handedness in data.handedness[0]
      ]
      hand_data_result.append(
          HandData(
              hand=hand_landmarks,
              world_hand=hand_world_landmarks,
              handedness=handedness_scores))
  return hand_data_result


class Dataset(classification_dataset.ClassificationDataset):
  """Dataset library for hand gesture recognizer."""

  @classmethod
  def from_folder(
      cls,
      dirname: str,
      hparams: Optional[HandDataPreprocessingParams] = None
  ) -> classification_dataset.ClassificationDataset:
    """Loads images and labels from the given directory.

    Directory contents are expected to be in the format:
    <root_dir>/<gesture_name>/*.jpg". One of the `gesture_name` must be `none`
    (case insensitive). The `none` sub-directory is expected to contain images
    of hands that don't belong to other gesture classes in <root_dir>. Assumes
    the image data of the same label are in the same subdirectory.

    Args:
      dirname: Name of the directory containing the data files.
      hparams: Optional hyperparameters for processing input hand gesture
        images.

    Returns:
      Dataset containing landmarks, labels, and other related info.

    Raises:
      ValueError: if the input data directory is empty or the label set does not
        contain label 'none' (case insensitive).
    """
    data_root = os.path.abspath(dirname)

    # Assumes the image data of the same label are in the same subdirectory,
    # gets image path and label names.
    all_image_paths = list(tf.io.gfile.glob(data_root + r'/*/*'))
    if not all_image_paths:
      raise ValueError('Image dataset directory is empty.')

    if not hparams:
      hparams = HandDataPreprocessingParams()

    if hparams.shuffle:
      # Random shuffle data.
      random.shuffle(all_image_paths)

    label_names = sorted(
        name for name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, name)))
    if 'none' not in [v.lower() for v in label_names]:
      raise ValueError('Label set does not contain label "None".')
    # Move label 'none' to the front of label list.
    none_idx = [v.lower() for v in label_names].index('none')
    none_value = label_names.pop(none_idx)
    label_names.insert(0, none_value)

    index_by_label = dict(
        (name, index) for index, name in enumerate(label_names))
    all_gesture_indices = [
        index_by_label[os.path.basename(os.path.dirname(path))]
        for path in all_image_paths
    ]

    # Compute hand data (including local hand landmark, world hand landmark, and
    # handedness) for all the input images.
    hand_data = _get_hand_data(
        all_image_paths=all_image_paths,
        min_detection_confidence=hparams.min_detection_confidence)

    # Get a list of the valid hand landmark sample in the hand data list.
    valid_indices = [
        i for i in range(len(hand_data)) if hand_data[i] is not None
    ]
    # Remove 'None' element from the hand data and label list.
    valid_hand_data = [dataclasses.asdict(hand_data[i]) for i in valid_indices]
    if not valid_hand_data:
      raise ValueError('No valid hand is detected.')

    valid_label = [all_gesture_indices[i] for i in valid_indices]

    # Convert list of dictionaries to dictionary of lists.
    hand_data_dict = {
        k: [lm[k] for lm in valid_hand_data] for k in valid_hand_data[0]
    }
    hand_ds = tf.data.Dataset.from_tensor_slices(hand_data_dict)

    embedder_model = model_util.load_keras_model(
        constants.GESTURE_EMBEDDER_KERAS_MODEL_FILES.get_path()
    )

    hand_ds = hand_ds.batch(batch_size=1)
    hand_embedding_ds = hand_ds.map(
        map_func=lambda feature: embedder_model(dict(feature)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    hand_embedding_ds = hand_embedding_ds.unbatch()

    # Create label dataset
    label_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(valid_label, tf.int64))

    label_one_hot_ds = label_ds.map(
        map_func=lambda index: tf.one_hot(index, len(label_names)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Create a dataset with (hand_embedding, one_hot_label) pairs
    hand_embedding_label_ds = tf.data.Dataset.zip(
        (hand_embedding_ds, label_one_hot_ds))

    tf.compat.v1.logging.info(
        'Load valid hands with size: {}, num_label: {}, labels: {}.'.format(
            len(valid_hand_data), len(label_names), ','.join(label_names)))
    return Dataset(
        dataset=hand_embedding_label_ds,
        label_names=label_names,
        size=len(valid_hand_data),
    )
