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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from google.protobuf import text_format
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.tasks.cc.vision.holistic_landmarker.proto import holistic_result_pb2
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.python.vision import holistic_landmarker
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module


HolisticLandmarkerResult = holistic_landmarker.HolisticLandmarkerResult
_HolisticResultProto = holistic_result_pb2.HolisticResult
_BaseOptions = base_options_module.BaseOptions
_Image = image_module.Image
_HolisticLandmarker = holistic_landmarker.HolisticLandmarker
_HolisticLandmarkerOptions = holistic_landmarker.HolisticLandmarkerOptions
_RUNNING_MODE = running_mode_module.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions

_HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE = 'holistic_landmarker.task'
_POSE_IMAGE = 'male_full_height_hands.jpg'
_CAT_IMAGE = 'cat.jpg'
_EXPECTED_HOLISTIC_RESULT = 'male_full_height_hands_result_cpu.pbtxt'
_IMAGE_WIDTH = 638
_IMAGE_HEIGHT = 1000
_LANDMARKS_MARGIN = 0.03
_BLENDSHAPES_MARGIN = 0.13
_VIDEO_LANDMARKS_MARGIN = 0.03
_VIDEO_BLENDSHAPES_MARGIN = 0.31
_LIVE_STREAM_LANDMARKS_MARGIN = 0.03
_LIVE_STREAM_BLENDSHAPES_MARGIN = 0.31


def _get_expected_holistic_landmarker_result(
    file_path: str,
) -> HolisticLandmarkerResult:
  holistic_result_file_path = test_utils.get_test_data_path(file_path)
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
    # Expects to have the same number of landmarks detected.
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
      self.assertEqual(
          elem.category_name, expected_blendshapes[i].category_name
      )
      self.assertAlmostEqual(
          elem.score,
          expected_blendshapes[i].score,
          delta=margin,
      )

  def _expect_holistic_landmarker_results_correct(
      self,
      actual_result: HolisticLandmarkerResult,
      expected_result: HolisticLandmarkerResult,
      output_segmentation_mask: bool,
      landmarks_margin: float,
      blendshapes_margin: float,
  ):
    self._expect_landmarks_correct(
        actual_result.pose_landmarks,
        expected_result.pose_landmarks,
        landmarks_margin,
    )
    self._expect_landmarks_correct(
        actual_result.face_landmarks,
        expected_result.face_landmarks,
        landmarks_margin,
    )
    self._expect_blendshapes_correct(
        actual_result.face_blendshapes,
        expected_result.face_blendshapes,
        blendshapes_margin,
    )
    if output_segmentation_mask:
      self.assertIsInstance(actual_result.segmentation_mask, _Image)
      self.assertEqual(actual_result.segmentation_mask.width, _IMAGE_WIDTH)
      self.assertEqual(actual_result.segmentation_mask.height, _IMAGE_HEIGHT)
    else:
      self.assertIsNone(actual_result.segmentation_mask)

  def test_create_from_file_succeeds_with_valid_model_path(self):
    # Creates with default option and valid model file successfully.
    with _HolisticLandmarker.create_from_model_path(
        self.model_path
    ) as landmarker:
      self.assertIsInstance(landmarker, _HolisticLandmarker)

  def test_create_from_options_succeeds_with_valid_model_path(self):
    # Creates with options containing model file successfully.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _HolisticLandmarkerOptions(base_options=base_options)
    with _HolisticLandmarker.create_from_options(options) as landmarker:
      self.assertIsInstance(landmarker, _HolisticLandmarker)

  def test_create_from_options_fails_with_invalid_model_path(self):
    # Invalid empty model path.
    with self.assertRaisesRegex(
        RuntimeError, 'Unable to open file at /path/to/invalid/model.tflite'
    ):
      base_options = _BaseOptions(
          model_asset_path='/path/to/invalid/model.tflite'
      )
      options = _HolisticLandmarkerOptions(base_options=base_options)
      _HolisticLandmarker.create_from_options(options)

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.model_path, 'rb') as f:
      base_options = _BaseOptions(model_asset_buffer=f.read())
      options = _HolisticLandmarkerOptions(base_options=base_options)
      landmarker = _HolisticLandmarker.create_from_options(options)
      self.assertIsInstance(landmarker, _HolisticLandmarker)

  @parameterized.parameters(
      (
          ModelFileType.FILE_NAME,
          _HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE,
          False,
          _get_expected_holistic_landmarker_result(_EXPECTED_HOLISTIC_RESULT),
      ),
      (
          ModelFileType.FILE_CONTENT,
          _HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE,
          False,
          _get_expected_holistic_landmarker_result(_EXPECTED_HOLISTIC_RESULT),
      ),
      (
          ModelFileType.FILE_NAME,
          _HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE,
          True,
          _get_expected_holistic_landmarker_result(_EXPECTED_HOLISTIC_RESULT),
      ),
      (
          ModelFileType.FILE_CONTENT,
          _HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE,
          True,
          _get_expected_holistic_landmarker_result(_EXPECTED_HOLISTIC_RESULT),
      ),
  )
  def test_detect(
      self,
      model_file_type,
      model_name,
      output_segmentation_mask,
      expected_holistic_landmarker_result,
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
        if expected_holistic_landmarker_result.face_blendshapes
        else False,
        output_segmentation_mask=output_segmentation_mask,
    )
    landmarker = _HolisticLandmarker.create_from_options(options)

    # Performs holistic landmarks detection on the input.
    detection_result = landmarker.detect(self.test_image)
    self._expect_holistic_landmarker_results_correct(
        detection_result,
        expected_holistic_landmarker_result,
        output_segmentation_mask,
        _LANDMARKS_MARGIN,
        _BLENDSHAPES_MARGIN,
    )
    # Closes the holistic landmarker explicitly when the holistic landmarker is
    # not used in a context.
    landmarker.close()

  @parameterized.parameters(
      (
          ModelFileType.FILE_NAME,
          _HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE,
          False,
          _get_expected_holistic_landmarker_result(_EXPECTED_HOLISTIC_RESULT),
      ),
      (
          ModelFileType.FILE_CONTENT,
          _HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE,
          True,
          _get_expected_holistic_landmarker_result(_EXPECTED_HOLISTIC_RESULT),
      ),
  )
  def test_detect_in_context(
      self,
      model_file_type,
      model_name,
      output_segmentation_mask,
      expected_holistic_landmarker_result,
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
        if expected_holistic_landmarker_result.face_blendshapes
        else False,
        output_segmentation_mask=output_segmentation_mask,
    )

    with _HolisticLandmarker.create_from_options(options) as landmarker:
      # Performs holistic landmarks detection on the input.
      detection_result = landmarker.detect(self.test_image)
      self._expect_holistic_landmarker_results_correct(
          detection_result,
          expected_holistic_landmarker_result,
          output_segmentation_mask,
          _LANDMARKS_MARGIN,
          _BLENDSHAPES_MARGIN,
      )

  def test_empty_detection_outputs(self):
    options = _HolisticLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path)
    )
    with _HolisticLandmarker.create_from_options(options) as landmarker:
      # Load the cat image.
      cat_test_image = _Image.create_from_file(
          test_utils.get_test_data_path(_CAT_IMAGE)
      )
      # Performs holistic landmarks detection on the input.
      detection_result = landmarker.detect(cat_test_image)
      self.assertEmpty(detection_result.face_landmarks)
      self.assertEmpty(detection_result.pose_landmarks)
      self.assertEmpty(detection_result.pose_world_landmarks)
      self.assertEmpty(detection_result.left_hand_landmarks)
      self.assertEmpty(detection_result.left_hand_world_landmarks)
      self.assertEmpty(detection_result.right_hand_landmarks)
      self.assertEmpty(detection_result.right_hand_world_landmarks)
      self.assertIsNone(detection_result.face_blendshapes)
      self.assertIsNone(detection_result.segmentation_mask)

  def test_missing_result_callback(self):
    options = _HolisticLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
    )
    with self.assertRaisesRegex(
        ValueError, r'result callback must be provided'
    ):
      with _HolisticLandmarker.create_from_options(
          options
      ) as unused_landmarker:
        pass

  @parameterized.parameters((_RUNNING_MODE.IMAGE), (_RUNNING_MODE.VIDEO))
  def test_illegal_result_callback(self, running_mode):
    options = _HolisticLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=running_mode,
        result_callback=mock.MagicMock(),
    )
    with self.assertRaisesRegex(
        ValueError, r'result callback should not be provided'
    ):
      with _HolisticLandmarker.create_from_options(
          options
      ) as unused_landmarker:
        pass

  def test_calling_detect_for_video_in_image_mode(self):
    options = _HolisticLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.IMAGE,
    )
    with _HolisticLandmarker.create_from_options(options) as landmarker:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the video mode'
      ):
        landmarker.detect_for_video(self.test_image, 0)

  def test_calling_detect_async_in_image_mode(self):
    options = _HolisticLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.IMAGE,
    )
    with _HolisticLandmarker.create_from_options(options) as landmarker:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the live stream mode'
      ):
        landmarker.detect_async(self.test_image, 0)

  def test_calling_detect_in_video_mode(self):
    options = _HolisticLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO,
    )
    with _HolisticLandmarker.create_from_options(options) as landmarker:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the image mode'
      ):
        landmarker.detect(self.test_image)

  def test_calling_detect_async_in_video_mode(self):
    options = _HolisticLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO,
    )
    with _HolisticLandmarker.create_from_options(options) as landmarker:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the live stream mode'
      ):
        landmarker.detect_async(self.test_image, 0)

  def test_detect_for_video_with_out_of_order_timestamp(self):
    options = _HolisticLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO,
    )
    with _HolisticLandmarker.create_from_options(options) as landmarker:
      unused_result = landmarker.detect_for_video(self.test_image, 1)
      with self.assertRaisesRegex(
          ValueError, r'Input timestamp must be monotonically increasing'
      ):
        landmarker.detect_for_video(self.test_image, 0)

  @parameterized.parameters(
      (
          _HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE,
          False,
          _get_expected_holistic_landmarker_result(_EXPECTED_HOLISTIC_RESULT),
      ),
      (
          _HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE,
          True,
          _get_expected_holistic_landmarker_result(_EXPECTED_HOLISTIC_RESULT),
      ),
  )
  def test_detect_for_video(
      self,
      model_name,
      output_segmentation_mask,
      expected_holistic_landmarker_result,
  ):
    # Creates holistic landmarker.
    model_path = test_utils.get_test_data_path(model_name)
    base_options = _BaseOptions(model_asset_path=model_path)
    options = _HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=_RUNNING_MODE.VIDEO,
        output_face_blendshapes=True
        if expected_holistic_landmarker_result.face_blendshapes
        else False,
        output_segmentation_mask=output_segmentation_mask,
    )

    with _HolisticLandmarker.create_from_options(options) as landmarker:
      for timestamp in range(0, 300, 30):
        # Performs holistic landmarks detection on the input.
        detection_result = landmarker.detect_for_video(
            self.test_image, timestamp
        )
        # Comparing results.
        self._expect_holistic_landmarker_results_correct(
            detection_result,
            expected_holistic_landmarker_result,
            output_segmentation_mask,
            _VIDEO_LANDMARKS_MARGIN,
            _VIDEO_BLENDSHAPES_MARGIN,
        )

  def test_calling_detect_in_live_stream_mode(self):
    options = _HolisticLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=mock.MagicMock(),
    )
    with _HolisticLandmarker.create_from_options(options) as landmarker:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the image mode'
      ):
        landmarker.detect(self.test_image)

  def test_calling_detect_for_video_in_live_stream_mode(self):
    options = _HolisticLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=mock.MagicMock(),
    )
    with _HolisticLandmarker.create_from_options(options) as landmarker:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the video mode'
      ):
        landmarker.detect_for_video(self.test_image, 0)

  def test_detect_async_calls_with_illegal_timestamp(self):
    options = _HolisticLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=mock.MagicMock(),
    )
    with _HolisticLandmarker.create_from_options(options) as landmarker:
      landmarker.detect_async(self.test_image, 100)
      with self.assertRaisesRegex(
          ValueError, r'Input timestamp must be monotonically increasing'
      ):
        landmarker.detect_async(self.test_image, 0)

  @parameterized.parameters(
      (
          _POSE_IMAGE,
          _HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE,
          False,
          _get_expected_holistic_landmarker_result(_EXPECTED_HOLISTIC_RESULT),
      ),
      (
          _POSE_IMAGE,
          _HOLISTIC_LANDMARKER_BUNDLE_ASSET_FILE,
          True,
          _get_expected_holistic_landmarker_result(_EXPECTED_HOLISTIC_RESULT),
      ),
  )
  def test_detect_async_calls(
      self,
      image_path,
      model_name,
      output_segmentation_mask,
      expected_holistic_landmarker_result,
  ):
    test_image = _Image.create_from_file(
        test_utils.get_test_data_path(image_path)
    )
    observed_timestamp_ms = -1

    def check_result(
        result: HolisticLandmarkerResult,
        output_image: _Image,
        timestamp_ms: int,
    ):
      # Comparing results.
      self._expect_holistic_landmarker_results_correct(
          result,
          expected_holistic_landmarker_result,
          output_segmentation_mask,
          _LIVE_STREAM_LANDMARKS_MARGIN,
          _LIVE_STREAM_BLENDSHAPES_MARGIN,
      )
      self.assertTrue(
          np.array_equal(output_image.numpy_view(), test_image.numpy_view())
      )
      self.assertLess(observed_timestamp_ms, timestamp_ms)
      self.observed_timestamp_ms = timestamp_ms

    model_path = test_utils.get_test_data_path(model_name)
    options = _HolisticLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        output_face_blendshapes=True
        if expected_holistic_landmarker_result.face_blendshapes
        else False,
        output_segmentation_mask=output_segmentation_mask,
        result_callback=check_result,
    )
    with _HolisticLandmarker.create_from_options(options) as landmarker:
      for timestamp in range(0, 300, 30):
        landmarker.detect_async(test_image, timestamp)


if __name__ == '__main__':
  absltest.main()
