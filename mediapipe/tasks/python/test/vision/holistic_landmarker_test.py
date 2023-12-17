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
"""Tests for holistic landmarker."""

import enum
from typing import List
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from google.protobuf import text_format
from mediapipe.framework.formats import classification_pb2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.cc.vision.holistic_landmarker.proto import holistic_result_pb2
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.components.containers import landmark as landmark_module
from mediapipe.tasks.python.components.containers import rect as rect_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.python.vision import holistic_landmarker
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module


HolisticLandmarkerResult = holistic_landmarker.HolisticLandmarkerResult
_HolisticResultProto = holistic_result_pb2.HolisticResult
_BaseOptions = base_options_module.BaseOptions
_Category = category_module.Category
_Rect = rect_module.Rect
_Landmark = landmark_module.Landmark
_NormalizedLandmark = landmark_module.NormalizedLandmark
_Image = image_module.Image
_HolisticLandmarker = holistic_landmarker.HolisticLandmarker
_HolisticLandmarkerOptions = holistic_landmarker.HolisticLandmarkerOptions
_RUNNING_MODE = running_mode_module.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions

_HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE = 'holistic_landmarker.task'
_POSE_IMAGE = 'male_full_height_hands.jpg'
_CAT_IMAGE = 'cat.jpg'
_EXPECTED_HOLISTIC_RESULT = "male_full_height_hands_result_cpu.pbtxt"
_LANDMARKS_MARGIN = 0.03
_BLENDSHAPES_MARGIN = 0.13


def _get_expected_holistic_landmarker_result(
    file_path: str,
) -> HolisticLandmarkerResult:
  holistic_result_file_path = test_utils.get_test_data_path(
    file_path
  )
  with open(holistic_result_file_path, 'rb') as f:
    holistic_result_proto = _HolisticResultProto()
    # Use this if a .pb file is available.
    # holistic_result_proto.ParseFromString(f.read())
    text_format.Parse(f.read(), holistic_result_proto)
    holistic_landmarker_result = HolisticLandmarkerResult.create_from_pb2(
      holistic_result_proto
    )
  return holistic_landmarker_result


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class HolisticLandmarkerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_image = _Image.create_from_file(
        test_utils.get_test_data_path(_POSE_IMAGE)
    )
    self.model_path = test_utils.get_test_data_path(
        _HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE
    )

  def _expect_landmarks_correct(
      self, actual_landmarks, expected_landmarks, margin
  ):
    # Expects to have the same number of poses detected.
    self.assertLen(actual_landmarks, len(expected_landmarks))

    for i, elem in enumerate(actual_landmarks):
      self.assertAlmostEqual(elem.x, expected_landmarks[i].x, delta=margin)
      self.assertAlmostEqual(elem.y, expected_landmarks[i].y, delta=margin)

  def _expect_blendshapes_correct(
      self, actual_blendshapes, expected_blendshapes, margin
  ):
    # Expects to have the same number of blendshapes.
    self.assertLen(actual_blendshapes, len(expected_blendshapes))

    for i, elem in enumerate(actual_blendshapes):
      self.assertEqual(elem.index, expected_blendshapes[i].index)
      self.assertAlmostEqual(
        elem.score,
        expected_blendshapes[i].score,
        delta=margin,
      )

  def _expect_holistic_landmarker_results_correct(
      self,
      actual_result: HolisticLandmarkerResult,
      expected_result: HolisticLandmarkerResult,
      output_segmentation_masks: bool,
      landmarks_margin: float,
      blendshapes_margin: float,
  ):
    self._expect_landmarks_correct(
      actual_result.pose_landmarks, expected_result.pose_landmarks,
      landmarks_margin
    )
    self._expect_landmarks_correct(
      actual_result.face_landmarks, expected_result.face_landmarks,
      landmarks_margin
    )
    self._expect_blendshapes_correct(
      actual_result.face_blendshapes, expected_result.face_blendshapes,
      blendshapes_margin
    )
    if output_segmentation_masks:
      self.assertIsInstance(actual_result.segmentation_masks, List)
      for _, mask in enumerate(actual_result.segmentation_masks):
        self.assertIsInstance(mask, _Image)
    else:
      self.assertIsNone(actual_result.segmentation_masks)

  @parameterized.parameters(
      (
          ModelFileType.FILE_NAME,
          _HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE,
          False,
          _get_expected_holistic_landmarker_result(_EXPECTED_HOLISTIC_RESULT)
      ),
      (
          ModelFileType.FILE_CONTENT,
          _HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE,
          False,
          _get_expected_holistic_landmarker_result(_EXPECTED_HOLISTIC_RESULT)
      ),
  )
  def test_detect(
      self,
      model_file_type,
      model_name,
      output_segmentation_masks,
      expected_holistic_landmarker_result: HolisticLandmarkerResult
  ):
    # Creates holistic landmarker.
    model_path = test_utils.get_test_data_path(model_name)
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(model_asset_path=model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(model_asset_buffer=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    options = _HolisticLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True
        if expected_holistic_landmarker_result.face_blendshapes else False,
        output_segmentation_masks=output_segmentation_masks,
    )
    landmarker = _HolisticLandmarker.create_from_options(options)

    # Performs holistic landmarks detection on the input.
    detection_result = landmarker.detect(self.test_image)
    self._expect_holistic_landmarker_results_correct(
        detection_result, expected_holistic_landmarker_result,
        output_segmentation_masks, _LANDMARKS_MARGIN, _BLENDSHAPES_MARGIN
    )
    # Closes the holistic landmarker explicitly when the holistic landmarker is
    # not used in a context.
    landmarker.close()


if __name__ == '__main__':
  absltest.main()
