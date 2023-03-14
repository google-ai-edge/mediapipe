# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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
"""Tests for face landmarker."""

import enum
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from google.protobuf import text_format
from mediapipe.framework.formats import landmark_pb2
from mediapipe.framework.formats import classification_pb2
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.components.containers import landmark as landmark_module
from mediapipe.tasks.python.components.containers import matrix_data as matrix_data_module
from mediapipe.tasks.python.components.containers import rect as rect_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.python.vision import face_landmarker
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

FaceLandmarkerResult = face_landmarker.FaceLandmarkerResult
_BaseOptions = base_options_module.BaseOptions
_Category = category_module.Category
_Rect = rect_module.Rect
_Landmark = landmark_module.Landmark
_MatrixData = matrix_data_module.MatrixData
_NormalizedLandmark = landmark_module.NormalizedLandmark
_Image = image_module.Image
_FaceLandmarker = face_landmarker.FaceLandmarker
_FaceLandmarkerOptions = face_landmarker.FaceLandmarkerOptions
_RUNNING_MODE = running_mode_module.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions

_FACE_LANDMARKER_BUNDLE_ASSET_FILE = 'face_landmarker.task'
_FACE_LANDMARKER_WITH_BLENDSHAPES_BUNDLE_ASSET_FILE = 'face_landmarker_with_blendshapes.task'
_PORTRAIT_IMAGE = 'portrait.jpg'
_PORTRAIT_EXPECTED_FACE_LANDMARKS = 'portrait_expected_face_landmarks.pbtxt'
_PORTRAIT_EXPECTED_FACE_LANDMARKS_WITH_ATTENTION = 'portrait_expected_face_landmarks_with_attention.pbtxt'
_PORTRAIT_EXPECTED_BLENDSHAPES = 'portrait_expected_blendshapes_with_attention.pbtxt'
_PORTRAIT_EXPECTED_FACE_GEOMETRY = 'portrait_expected_face_geometry_with_attention.pbtxt'
_LANDMARKS_DIFF_MARGIN = 0.03
_BLENDSHAPES_DIFF_MARGIN = 0.1
_FACIAL_TRANSFORMATION_MATRIX_DIFF_MARGIN = 0.02


def _get_expected_face_landmarks(file_path: str):
  proto_file_path = test_utils.get_test_data_path(file_path)
  with open(proto_file_path, 'rb') as f:
    proto = landmark_pb2.NormalizedLandmarkList()
    text_format.Parse(f.read(), proto)
    face_landmarks = []
    for landmark in proto.landmark:
      face_landmarks.append(_NormalizedLandmark.create_from_pb2(landmark))
  return face_landmarks


def _get_expected_face_blendshapes(file_path: str):
  proto_file_path = test_utils.get_test_data_path(file_path)
  with open(proto_file_path, 'rb') as f:
    proto = classification_pb2.ClassificationList()
    text_format.Parse(f.read(), proto)
    face_blendshapes_categories = []
    face_blendshapes_classifications = classification_pb2.ClassificationList()
    face_blendshapes_classifications.MergeFrom(proto)
    for face_blendshapes in face_blendshapes_classifications.classification:
      face_blendshapes_categories.append(
        category_module.Category(
          index=face_blendshapes.index,
          score=face_blendshapes.score,
          display_name=face_blendshapes.display_name,
          category_name=face_blendshapes.label))
  return face_blendshapes_categories


def _make_expected_facial_transformation_matrixes():
  data = np.array([[0.9995292, -0.005092691, 0.030254554, -0.37340546],
          [0.0072318087, 0.99744856, -0.07102106, 22.212194],
          [-0.029815676, 0.07120642, 0.9970159, -64.76358],
          [0, 0, 0, 1]])
  rows, cols = len(data), len(data[0])
  facial_transformation_matrixes_results = []
  facial_transformation_matrix = _MatrixData(rows, cols, data)
  facial_transformation_matrixes_results.append(facial_transformation_matrix)
  return facial_transformation_matrixes_results


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class HandLandmarkerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_image = _Image.create_from_file(
        test_utils.get_test_data_path(_PORTRAIT_IMAGE))
    self.model_path = test_utils.get_test_data_path(
        _FACE_LANDMARKER_BUNDLE_ASSET_FILE)

  def _expect_landmarks_correct(self, actual_landmarks, expected_landmarks):
    # Expects to have the same number of faces detected.
    self.assertLen(actual_landmarks, len(expected_landmarks))

    for i, rename_me in enumerate(actual_landmarks):
      self.assertAlmostEqual(
        rename_me.x,
        expected_landmarks[i].x,
        delta=_LANDMARKS_DIFF_MARGIN)
      self.assertAlmostEqual(
        rename_me.y,
        expected_landmarks[i].y,
        delta=_LANDMARKS_DIFF_MARGIN)

  def _expect_blendshapes_correct(self, actual_blendshapes, expected_blendshapes):
    # Expects to have the same number of blendshapes.
    self.assertLen(actual_blendshapes, len(expected_blendshapes))

    for i, rename_me in enumerate(actual_blendshapes):
      self.assertEqual(rename_me.index, expected_blendshapes[i].index)
      self.assertAlmostEqual(
        rename_me.score,
        expected_blendshapes[i].score,
        delta=_BLENDSHAPES_DIFF_MARGIN)

  def _expect_facial_transformation_matrix_correct(self, actual_matrix_list,
                                                   expected_matrix_list):
    self.assertLen(actual_matrix_list, len(expected_matrix_list))

    for i, rename_me in enumerate(actual_matrix_list):
      self.assertEqual(rename_me.rows, expected_matrix_list[i].rows)
      self.assertEqual(rename_me.cols, expected_matrix_list[i].cols)
      self.assertAlmostEqual(
        rename_me.data,
        expected_matrix_list[i].data,
        delta=_FACIAL_TRANSFORMATION_MATRIX_DIFF_MARGIN)

  def test_create_from_file_succeeds_with_valid_model_path(self):
    # Creates with default option and valid model file successfully.
    with _FaceLandmarker.create_from_model_path(self.model_path) as landmarker:
      self.assertIsInstance(landmarker, _FaceLandmarker)

  def test_create_from_options_succeeds_with_valid_model_path(self):
    # Creates with options containing model file successfully.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _FaceLandmarkerOptions(base_options=base_options)
    with _FaceLandmarker.create_from_options(options) as landmarker:
      self.assertIsInstance(landmarker, _FaceLandmarker)

  def test_create_from_options_fails_with_invalid_model_path(self):
    # Invalid empty model path.
    with self.assertRaisesRegex(
        RuntimeError, 'Unable to open file at /path/to/invalid/model.tflite'):
      base_options = _BaseOptions(
          model_asset_path='/path/to/invalid/model.tflite')
      options = _FaceLandmarkerOptions(base_options=base_options)
      _FaceLandmarker.create_from_options(options)

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.model_path, 'rb') as f:
      base_options = _BaseOptions(model_asset_buffer=f.read())
      options = _FaceLandmarkerOptions(base_options=base_options)
      landmarker = _FaceLandmarker.create_from_options(options)
      self.assertIsInstance(landmarker, _FaceLandmarker)

  @parameterized.parameters(
      (ModelFileType.FILE_NAME, _FACE_LANDMARKER_BUNDLE_ASSET_FILE,
      _get_expected_face_landmarks(
        _PORTRAIT_EXPECTED_FACE_LANDMARKS), None, None),
      (ModelFileType.FILE_CONTENT, _FACE_LANDMARKER_BUNDLE_ASSET_FILE,
       _get_expected_face_landmarks(
        _PORTRAIT_EXPECTED_FACE_LANDMARKS), None, None),
      (ModelFileType.FILE_NAME,
       _FACE_LANDMARKER_WITH_BLENDSHAPES_BUNDLE_ASSET_FILE,
       _get_expected_face_landmarks(
         _PORTRAIT_EXPECTED_FACE_LANDMARKS_WITH_ATTENTION), None, None),
      (ModelFileType.FILE_CONTENT,
       _FACE_LANDMARKER_WITH_BLENDSHAPES_BUNDLE_ASSET_FILE,
       _get_expected_face_landmarks(
         _PORTRAIT_EXPECTED_FACE_LANDMARKS_WITH_ATTENTION), None, None),
      (ModelFileType.FILE_NAME,
       _FACE_LANDMARKER_WITH_BLENDSHAPES_BUNDLE_ASSET_FILE,
       _get_expected_face_landmarks(
         _PORTRAIT_EXPECTED_FACE_LANDMARKS_WITH_ATTENTION),
       _get_expected_face_blendshapes(
         _PORTRAIT_EXPECTED_BLENDSHAPES), None),
      (ModelFileType.FILE_CONTENT,
       _FACE_LANDMARKER_WITH_BLENDSHAPES_BUNDLE_ASSET_FILE,
       _get_expected_face_landmarks(
         _PORTRAIT_EXPECTED_FACE_LANDMARKS_WITH_ATTENTION),
       _get_expected_face_blendshapes(
         _PORTRAIT_EXPECTED_BLENDSHAPES), None),
      # (ModelFileType.FILE_NAME,
      #  _FACE_LANDMARKER_WITH_BLENDSHAPES_BUNDLE_ASSET_FILE,
      #  _get_expected_face_landmarks(
      #    _PORTRAIT_EXPECTED_FACE_LANDMARKS_WITH_ATTENTION),
      #  _get_expected_face_blendshapes(
      #    _PORTRAIT_EXPECTED_BLENDSHAPES),
      #  _make_expected_facial_transformation_matrixes()),
      # (ModelFileType.FILE_CONTENT,
      #  _FACE_LANDMARKER_WITH_BLENDSHAPES_BUNDLE_ASSET_FILE,
      #  _get_expected_face_landmarks(
      #    _PORTRAIT_EXPECTED_FACE_LANDMARKS_WITH_ATTENTION),
      #  _get_expected_face_blendshapes(
      #    _PORTRAIT_EXPECTED_BLENDSHAPES),
      #  _make_expected_facial_transformation_matrixes())
  )
  def test_detect(self, model_file_type, model_name, expected_face_landmarks,
                  expected_face_blendshapes, expected_facial_transformation_matrix):
    # Creates face landmarker.
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

    options = _FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True if expected_face_blendshapes else False,
        output_facial_transformation_matrixes=True
        if expected_facial_transformation_matrix else False)
    landmarker = _FaceLandmarker.create_from_options(options)

    # Performs face landmarks detection on the input.
    detection_result = landmarker.detect(self.test_image)
    # Comparing results.
    if expected_face_landmarks is not None:
      self._expect_landmarks_correct(detection_result.face_landmarks[0],
                                     expected_face_landmarks)
    if expected_face_blendshapes is not None:
      self._expect_blendshapes_correct(detection_result.face_blendshapes[0],
                                       expected_face_blendshapes)
    if expected_facial_transformation_matrix is not None:
      self._expect_facial_transformation_matrix_correct(
          detection_result.facial_transformation_matrixes[0],
          expected_facial_transformation_matrix)

    # Closes the face landmarker explicitly when the face landmarker is not used
    # in a context.
    landmarker.close()


if __name__ == '__main__':
  absltest.main()
