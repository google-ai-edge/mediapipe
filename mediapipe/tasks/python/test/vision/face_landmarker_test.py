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
"""Tests for face landmarker."""

import enum
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from google.protobuf import text_format
from mediapipe.framework.formats import classification_pb2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.components.containers import landmark as landmark_module
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
_NormalizedLandmark = landmark_module.NormalizedLandmark
_Image = image_module.Image
_FaceLandmarker = face_landmarker.FaceLandmarker
_FaceLandmarkerOptions = face_landmarker.FaceLandmarkerOptions
_RUNNING_MODE = running_mode_module.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions

_FACE_LANDMARKER_BUNDLE_ASSET_FILE = 'face_landmarker_v2.task'
_PORTRAIT_IMAGE = 'portrait.jpg'
_CAT_IMAGE = 'cat.jpg'
_PORTRAIT_EXPECTED_FACE_LANDMARKS = 'portrait_expected_face_landmarks.pbtxt'
_PORTRAIT_EXPECTED_BLENDSHAPES = 'portrait_expected_blendshapes.pbtxt'
_LANDMARKS_MARGIN = 0.03
_BLENDSHAPES_MARGIN = 0.13
_FACIAL_TRANSFORMATION_MATRIX_MARGIN = 0.02


def _get_expected_face_landmarks(file_path: str):
  proto_file_path = test_utils.get_test_data_path(file_path)
  face_landmarks_results = []
  with open(proto_file_path, 'rb') as f:
    proto = landmark_pb2.NormalizedLandmarkList()
    text_format.Parse(f.read(), proto)
    face_landmarks = []
    for landmark in proto.landmark:
      face_landmarks.append(_NormalizedLandmark.create_from_pb2(landmark))
  face_landmarks_results.append(face_landmarks)
  return face_landmarks_results


def _get_expected_face_blendshapes(file_path: str):
  proto_file_path = test_utils.get_test_data_path(file_path)
  face_blendshapes_results = []
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
              category_name=face_blendshapes.label,
          )
      )
  face_blendshapes_results.append(face_blendshapes_categories)
  return face_blendshapes_results


def _get_expected_facial_transformation_matrixes():
  matrix = np.array([
      [0.9995292, -0.01294756, 0.038823195, -0.3691378],
      [0.0072318087, 0.9937692, -0.1101321, 22.75809],
      [-0.03715533, 0.11070588, 0.99315894, -65.765925],
      [0, 0, 0, 1],
  ])
  facial_transformation_matrixes_results = []
  facial_transformation_matrixes_results.append(matrix)
  return facial_transformation_matrixes_results


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class FaceLandmarkerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_image = _Image.create_from_file(
        test_utils.get_test_data_path(_PORTRAIT_IMAGE)
    )
    self.model_path = test_utils.get_test_data_path(
        _FACE_LANDMARKER_BUNDLE_ASSET_FILE
    )

  def _expect_landmarks_correct(self, actual_landmarks, expected_landmarks):
    # Expects to have the same number of faces detected.
    self.assertLen(actual_landmarks, len(expected_landmarks))

    for i, _ in enumerate(actual_landmarks):
      for j, elem in enumerate(actual_landmarks[i]):
        self.assertAlmostEqual(
            elem.x, expected_landmarks[i][j].x, delta=_LANDMARKS_MARGIN
        )
        self.assertAlmostEqual(
            elem.y, expected_landmarks[i][j].y, delta=_LANDMARKS_MARGIN
        )

  def _expect_blendshapes_correct(
      self, actual_blendshapes, expected_blendshapes
  ):
    # Expects to have the same number of blendshapes.
    self.assertLen(actual_blendshapes, len(expected_blendshapes))

    for i, _ in enumerate(actual_blendshapes):
      for j, elem in enumerate(actual_blendshapes[i]):
        self.assertEqual(elem.index, expected_blendshapes[i][j].index)
        self.assertAlmostEqual(
            elem.score,
            expected_blendshapes[i][j].score,
            delta=_BLENDSHAPES_MARGIN,
        )

  def _expect_facial_transformation_matrixes_correct(
      self, actual_matrix_list, expected_matrix_list
  ):
    self.assertLen(actual_matrix_list, len(expected_matrix_list))

    for i, elem in enumerate(actual_matrix_list):
      self.assertEqual(elem.shape[0], expected_matrix_list[i].shape[0])
      self.assertEqual(elem.shape[1], expected_matrix_list[i].shape[1])
      self.assertSequenceAlmostEqual(
          elem.flatten(),
          expected_matrix_list[i].flatten(),
          delta=_FACIAL_TRANSFORMATION_MATRIX_MARGIN,
      )

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
        RuntimeError, 'Unable to open file at /path/to/invalid/model.tflite'
    ):
      base_options = _BaseOptions(
          model_asset_path='/path/to/invalid/model.tflite'
      )
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
      (
          ModelFileType.FILE_NAME,
          _FACE_LANDMARKER_BUNDLE_ASSET_FILE,
          _get_expected_face_landmarks(_PORTRAIT_EXPECTED_FACE_LANDMARKS),
          None,
          None,
      ),
      (
          ModelFileType.FILE_CONTENT,
          _FACE_LANDMARKER_BUNDLE_ASSET_FILE,
          _get_expected_face_landmarks(_PORTRAIT_EXPECTED_FACE_LANDMARKS),
          None,
          None,
      ),
  )
  def test_detect(
      self,
      model_file_type,
      model_name,
      expected_face_landmarks,
      expected_face_blendshapes,
      expected_facial_transformation_matrixes,
  ):
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
        if expected_facial_transformation_matrixes
        else False,
    )
    landmarker = _FaceLandmarker.create_from_options(options)

    # Performs face landmarks detection on the input.
    detection_result = landmarker.detect(self.test_image)
    # Comparing results.
    if expected_face_landmarks is not None:
      self._expect_landmarks_correct(
          detection_result.face_landmarks, expected_face_landmarks
      )
    if expected_face_blendshapes is not None:
      self._expect_blendshapes_correct(
          detection_result.face_blendshapes, expected_face_blendshapes
      )
    if expected_facial_transformation_matrixes is not None:
      self._expect_facial_transformation_matrixes_correct(
          detection_result.facial_transformation_matrixes,
          expected_facial_transformation_matrixes,
      )

    # Closes the face landmarker explicitly when the face landmarker is not used
    # in a context.
    landmarker.close()

  @parameterized.parameters(
      (
          ModelFileType.FILE_NAME,
          _FACE_LANDMARKER_BUNDLE_ASSET_FILE,
          _get_expected_face_landmarks(_PORTRAIT_EXPECTED_FACE_LANDMARKS),
          None,
          None,
      ),
      (
          ModelFileType.FILE_CONTENT,
          _FACE_LANDMARKER_BUNDLE_ASSET_FILE,
          _get_expected_face_landmarks(_PORTRAIT_EXPECTED_FACE_LANDMARKS),
          None,
          None,
      ),
  )
  def test_detect_in_context(
      self,
      model_file_type,
      model_name,
      expected_face_landmarks,
      expected_face_blendshapes,
      expected_facial_transformation_matrixes,
  ):
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
        if expected_facial_transformation_matrixes
        else False,
    )

    with _FaceLandmarker.create_from_options(options) as landmarker:
      # Performs face landmarks detection on the input.
      detection_result = landmarker.detect(self.test_image)
      # Comparing results.
      if expected_face_landmarks is not None:
        self._expect_landmarks_correct(
            detection_result.face_landmarks, expected_face_landmarks
        )
      if expected_face_blendshapes is not None:
        self._expect_blendshapes_correct(
            detection_result.face_blendshapes, expected_face_blendshapes
        )
      if expected_facial_transformation_matrixes is not None:
        self._expect_facial_transformation_matrixes_correct(
            detection_result.facial_transformation_matrixes,
            expected_facial_transformation_matrixes,
        )

  def test_empty_detection_outputs(self):
    options = _FaceLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path)
    )
    with _FaceLandmarker.create_from_options(options) as landmarker:
      # Load the image with no faces.
      no_faces_test_image = _Image.create_from_file(
          test_utils.get_test_data_path(_CAT_IMAGE)
      )
      # Performs face landmarks detection on the input.
      detection_result = landmarker.detect(no_faces_test_image)
      self.assertEmpty(detection_result.face_landmarks)
      self.assertEmpty(detection_result.face_blendshapes)
      self.assertEmpty(detection_result.facial_transformation_matrixes)

  def test_missing_result_callback(self):
    options = _FaceLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
    )
    with self.assertRaisesRegex(
        ValueError, r'result callback must be provided'
    ):
      with _FaceLandmarker.create_from_options(options) as unused_landmarker:
        pass

  @parameterized.parameters((_RUNNING_MODE.IMAGE), (_RUNNING_MODE.VIDEO))
  def test_illegal_result_callback(self, running_mode):
    options = _FaceLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=running_mode,
        result_callback=mock.MagicMock(),
    )
    with self.assertRaisesRegex(
        ValueError, r'result callback should not be provided'
    ):
      with _FaceLandmarker.create_from_options(options) as unused_landmarker:
        pass

  def test_calling_detect_for_video_in_image_mode(self):
    options = _FaceLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.IMAGE,
    )
    with _FaceLandmarker.create_from_options(options) as landmarker:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the video mode'
      ):
        landmarker.detect_for_video(self.test_image, 0)

  def test_calling_detect_async_in_image_mode(self):
    options = _FaceLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.IMAGE,
    )
    with _FaceLandmarker.create_from_options(options) as landmarker:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the live stream mode'
      ):
        landmarker.detect_async(self.test_image, 0)

  def test_calling_detect_in_video_mode(self):
    options = _FaceLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO,
    )
    with _FaceLandmarker.create_from_options(options) as landmarker:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the image mode'
      ):
        landmarker.detect(self.test_image)

  def test_calling_detect_async_in_video_mode(self):
    options = _FaceLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO,
    )
    with _FaceLandmarker.create_from_options(options) as landmarker:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the live stream mode'
      ):
        landmarker.detect_async(self.test_image, 0)

  def test_detect_for_video_with_out_of_order_timestamp(self):
    options = _FaceLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO,
    )
    with _FaceLandmarker.create_from_options(options) as landmarker:
      unused_result = landmarker.detect_for_video(self.test_image, 1)
      with self.assertRaisesRegex(
          ValueError, r'Input timestamp must be monotonically increasing'
      ):
        landmarker.detect_for_video(self.test_image, 0)

  @parameterized.parameters(
      (
          _FACE_LANDMARKER_BUNDLE_ASSET_FILE,
          _get_expected_face_landmarks(_PORTRAIT_EXPECTED_FACE_LANDMARKS),
          None,
          None,
      ),
  )
  def test_detect_for_video(
      self,
      model_name,
      expected_face_landmarks,
      expected_face_blendshapes,
      expected_facial_transformation_matrixes,
  ):
    # Creates face landmarker.
    model_path = test_utils.get_test_data_path(model_name)
    base_options = _BaseOptions(model_asset_path=model_path)

    options = _FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=_RUNNING_MODE.VIDEO,
        output_face_blendshapes=True if expected_face_blendshapes else False,
        output_facial_transformation_matrixes=True
        if expected_facial_transformation_matrixes
        else False,
    )

    with _FaceLandmarker.create_from_options(options) as landmarker:
      for timestamp in range(0, 300, 30):
        # Performs face landmarks detection on the input.
        detection_result = landmarker.detect_for_video(
            self.test_image, timestamp
        )
        # Comparing results.
        if expected_face_landmarks is not None:
          self._expect_landmarks_correct(
              detection_result.face_landmarks, expected_face_landmarks
          )
        if expected_face_blendshapes is not None:
          self._expect_blendshapes_correct(
              detection_result.face_blendshapes, expected_face_blendshapes
          )
        if expected_facial_transformation_matrixes is not None:
          self._expect_facial_transformation_matrixes_correct(
              detection_result.facial_transformation_matrixes,
              expected_facial_transformation_matrixes,
          )

  def test_calling_detect_in_live_stream_mode(self):
    options = _FaceLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=mock.MagicMock(),
    )
    with _FaceLandmarker.create_from_options(options) as landmarker:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the image mode'
      ):
        landmarker.detect(self.test_image)

  def test_calling_detect_for_video_in_live_stream_mode(self):
    options = _FaceLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=mock.MagicMock(),
    )
    with _FaceLandmarker.create_from_options(options) as landmarker:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the video mode'
      ):
        landmarker.detect_for_video(self.test_image, 0)

  def test_detect_async_calls_with_illegal_timestamp(self):
    options = _FaceLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=mock.MagicMock(),
    )
    with _FaceLandmarker.create_from_options(options) as landmarker:
      landmarker.detect_async(self.test_image, 100)
      with self.assertRaisesRegex(
          ValueError, r'Input timestamp must be monotonically increasing'
      ):
        landmarker.detect_async(self.test_image, 0)

  @parameterized.parameters(
      (
          _PORTRAIT_IMAGE,
          _FACE_LANDMARKER_BUNDLE_ASSET_FILE,
          _get_expected_face_landmarks(_PORTRAIT_EXPECTED_FACE_LANDMARKS),
          None,
          None,
      ),
  )
  def test_detect_async_calls(
      self,
      image_path,
      model_name,
      expected_face_landmarks,
      expected_face_blendshapes,
      expected_facial_transformation_matrixes,
  ):
    test_image = _Image.create_from_file(
        test_utils.get_test_data_path(image_path)
    )
    observed_timestamp_ms = -1

    def check_result(
        result: FaceLandmarkerResult, output_image: _Image, timestamp_ms: int
    ):
      # Comparing results.
      if expected_face_landmarks is not None:
        self._expect_landmarks_correct(
            result.face_landmarks, expected_face_landmarks
        )
      if expected_face_blendshapes is not None:
        self._expect_blendshapes_correct(
            result.face_blendshapes, expected_face_blendshapes
        )
      if expected_facial_transformation_matrixes is not None:
        self._expect_facial_transformation_matrixes_correct(
            result.facial_transformation_matrixes,
            expected_facial_transformation_matrixes,
        )
      self.assertTrue(
          np.array_equal(output_image.numpy_view(), test_image.numpy_view())
      )
      self.assertLess(observed_timestamp_ms, timestamp_ms)
      self.observed_timestamp_ms = timestamp_ms

    model_path = test_utils.get_test_data_path(model_name)
    options = _FaceLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        output_face_blendshapes=True if expected_face_blendshapes else False,
        output_facial_transformation_matrixes=True
        if expected_facial_transformation_matrixes
        else False,
        result_callback=check_result,
    )
    with _FaceLandmarker.create_from_options(options) as landmarker:
      for timestamp in range(0, 300, 30):
        landmarker.detect_async(test_image, timestamp)


if __name__ == '__main__':
  absltest.main()
