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
"""Tests for pose landmarker."""

import enum
from typing import List
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from google.protobuf import text_format
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.tasks.cc.components.containers.proto import landmarks_detection_result_pb2
from mediapipe.tasks.python.components.containers import landmark as landmark_module
from mediapipe.tasks.python.components.containers import landmark_detection_result as landmark_detection_result_module
from mediapipe.tasks.python.components.containers import rect as rect_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.python.vision import pose_landmarker
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

PoseLandmarkerResult = pose_landmarker.PoseLandmarkerResult
_LandmarksDetectionResultProto = (
    landmarks_detection_result_pb2.LandmarksDetectionResult
)
_BaseOptions = base_options_module.BaseOptions
_Rect = rect_module.Rect
_Landmark = landmark_module.Landmark
_NormalizedLandmark = landmark_module.NormalizedLandmark
_LandmarksDetectionResult = (
    landmark_detection_result_module.LandmarksDetectionResult
)
_Image = image_module.Image
_PoseLandmarker = pose_landmarker.PoseLandmarker
_PoseLandmarkerOptions = pose_landmarker.PoseLandmarkerOptions
_RUNNING_MODE = running_mode_module.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions

_POSE_LANDMARKER_BUNDLE_ASSET_FILE = 'pose_landmarker.task'
_BURGER_IMAGE = 'burger.jpg'
_POSE_IMAGE = 'pose.jpg'
_POSE_LANDMARKS = 'pose_landmarks.pbtxt'
_LANDMARKS_MARGIN = 0.03


def _get_expected_pose_landmarker_result(
    file_path: str,
) -> PoseLandmarkerResult:
  landmarks_detection_result_file_path = test_utils.get_test_data_path(
      file_path
  )
  with open(landmarks_detection_result_file_path, 'rb') as f:
    landmarks_detection_result_proto = _LandmarksDetectionResultProto()
    # Use this if a .pb file is available.
    # landmarks_detection_result_proto.ParseFromString(f.read())
    text_format.Parse(f.read(), landmarks_detection_result_proto)
    landmarks_detection_result = _LandmarksDetectionResult.create_from_pb2(
        landmarks_detection_result_proto
    )
  return PoseLandmarkerResult(
      pose_landmarks=[landmarks_detection_result.landmarks],
      pose_world_landmarks=[],
  )


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class PoseLandmarkerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_image = _Image.create_from_file(
        test_utils.get_test_data_path(_POSE_IMAGE)
    )
    self.model_path = test_utils.get_test_data_path(
        _POSE_LANDMARKER_BUNDLE_ASSET_FILE
    )

  def _expect_pose_landmarks_correct(
      self, actual_landmarks, expected_landmarks, margin
  ):
    # Expects to have the same number of poses detected.
    self.assertLen(actual_landmarks, len(expected_landmarks))

    for i, _ in enumerate(actual_landmarks):
      for j, elem in enumerate(actual_landmarks[i]):
        self.assertAlmostEqual(elem.x, expected_landmarks[i][j].x, delta=margin)
        self.assertAlmostEqual(elem.y, expected_landmarks[i][j].y, delta=margin)

  def _expect_pose_landmarker_results_correct(
      self,
      actual_result: PoseLandmarkerResult,
      expected_result: PoseLandmarkerResult,
      output_segmentation_masks: bool,
      margin: float,
  ):
    self._expect_pose_landmarks_correct(
        actual_result.pose_landmarks, expected_result.pose_landmarks, margin
    )
    if output_segmentation_masks:
      self.assertIsInstance(actual_result.segmentation_masks, List)
      for _, mask in enumerate(actual_result.segmentation_masks):
        self.assertIsInstance(mask, _Image)
    else:
      self.assertIsNone(actual_result.segmentation_masks)

  def test_create_from_file_succeeds_with_valid_model_path(self):
    # Creates with default option and valid model file successfully.
    with _PoseLandmarker.create_from_model_path(self.model_path) as landmarker:
      self.assertIsInstance(landmarker, _PoseLandmarker)

  def test_create_from_options_succeeds_with_valid_model_path(self):
    # Creates with options containing model file successfully.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _PoseLandmarkerOptions(base_options=base_options)
    with _PoseLandmarker.create_from_options(options) as landmarker:
      self.assertIsInstance(landmarker, _PoseLandmarker)

  def test_create_from_options_fails_with_invalid_model_path(self):
    # Invalid empty model path.
    with self.assertRaisesRegex(
        RuntimeError, 'Unable to open file at /path/to/invalid/model.tflite'
    ):
      base_options = _BaseOptions(
          model_asset_path='/path/to/invalid/model.tflite'
      )
      options = _PoseLandmarkerOptions(base_options=base_options)
      _PoseLandmarker.create_from_options(options)

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.model_path, 'rb') as f:
      base_options = _BaseOptions(model_asset_buffer=f.read())
      options = _PoseLandmarkerOptions(base_options=base_options)
      landmarker = _PoseLandmarker.create_from_options(options)
      self.assertIsInstance(landmarker, _PoseLandmarker)

  @parameterized.parameters(
      (
          ModelFileType.FILE_NAME,
          False,
          _get_expected_pose_landmarker_result(_POSE_LANDMARKS),
      ),
      (
          ModelFileType.FILE_CONTENT,
          False,
          _get_expected_pose_landmarker_result(_POSE_LANDMARKS),
      ),
      (
          ModelFileType.FILE_NAME,
          True,
          _get_expected_pose_landmarker_result(_POSE_LANDMARKS),
      ),
      (
          ModelFileType.FILE_CONTENT,
          True,
          _get_expected_pose_landmarker_result(_POSE_LANDMARKS),
      ),
  )
  def test_detect(
      self,
      model_file_type,
      output_segmentation_masks,
      expected_detection_result,
  ):
    # Creates pose landmarker.
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(model_asset_path=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(model_asset_buffer=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    options = _PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=output_segmentation_masks,
    )
    landmarker = _PoseLandmarker.create_from_options(options)

    # Performs pose landmarks detection on the input.
    detection_result = landmarker.detect(self.test_image)

    # Comparing results.
    self._expect_pose_landmarker_results_correct(
        detection_result,
        expected_detection_result,
        output_segmentation_masks,
        _LANDMARKS_MARGIN,
    )
    # Closes the pose landmarker explicitly when the pose landmarker is not used
    # in a context.
    landmarker.close()

  @parameterized.parameters(
      (
          ModelFileType.FILE_NAME,
          False,
          _get_expected_pose_landmarker_result(_POSE_LANDMARKS),
      ),
      (
          ModelFileType.FILE_CONTENT,
          False,
          _get_expected_pose_landmarker_result(_POSE_LANDMARKS),
      ),
      (
          ModelFileType.FILE_NAME,
          True,
          _get_expected_pose_landmarker_result(_POSE_LANDMARKS),
      ),
      (
          ModelFileType.FILE_CONTENT,
          True,
          _get_expected_pose_landmarker_result(_POSE_LANDMARKS),
      ),
  )
  def test_detect_in_context(
      self,
      model_file_type,
      output_segmentation_masks,
      expected_detection_result,
  ):
    # Creates pose landmarker.
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(model_asset_path=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(model_asset_buffer=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    options = _PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=output_segmentation_masks,
    )
    with _PoseLandmarker.create_from_options(options) as landmarker:
      # Performs pose landmarks detection on the input.
      detection_result = landmarker.detect(self.test_image)

      # Comparing results.
      self._expect_pose_landmarker_results_correct(
          detection_result,
          expected_detection_result,
          output_segmentation_masks,
          _LANDMARKS_MARGIN,
      )

  def test_detect_fails_with_region_of_interest(self):
    # Creates pose landmarker.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _PoseLandmarkerOptions(base_options=base_options)
    with self.assertRaisesRegex(
        ValueError, "This task doesn't support region-of-interest."
    ):
      with _PoseLandmarker.create_from_options(options) as landmarker:
        # Set the `region_of_interest` parameter using `ImageProcessingOptions`.
        image_processing_options = _ImageProcessingOptions(
            region_of_interest=_Rect(0, 0, 1, 1)
        )
        # Attempt to perform pose landmarks detection on the cropped input.
        landmarker.detect(self.test_image, image_processing_options)

  def test_empty_detection_outputs(self):
    # Creates pose landmarker.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _PoseLandmarkerOptions(base_options=base_options)
    with _PoseLandmarker.create_from_options(options) as landmarker:
      # Load an image with no poses.
      test_image = _Image.create_from_file(
          test_utils.get_test_data_path(_BURGER_IMAGE)
      )
      # Performs pose landmarks detection on the input.
      detection_result = landmarker.detect(test_image)
      # Comparing results.
      self.assertEmpty(detection_result.pose_landmarks)
      self.assertEmpty(detection_result.pose_world_landmarks)

  def test_missing_result_callback(self):
    options = _PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
    )
    with self.assertRaisesRegex(
        ValueError, r'result callback must be provided'
    ):
      with _PoseLandmarker.create_from_options(options) as unused_landmarker:
        pass

  @parameterized.parameters((_RUNNING_MODE.IMAGE), (_RUNNING_MODE.VIDEO))
  def test_illegal_result_callback(self, running_mode):
    options = _PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=running_mode,
        result_callback=mock.MagicMock(),
    )
    with self.assertRaisesRegex(
        ValueError, r'result callback should not be provided'
    ):
      with _PoseLandmarker.create_from_options(options) as unused_landmarker:
        pass

  def test_calling_detect_for_video_in_image_mode(self):
    options = _PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.IMAGE,
    )
    with _PoseLandmarker.create_from_options(options) as landmarker:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the video mode'
      ):
        landmarker.detect_for_video(self.test_image, 0)

  def test_calling_detect_async_in_image_mode(self):
    options = _PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.IMAGE,
    )
    with _PoseLandmarker.create_from_options(options) as landmarker:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the live stream mode'
      ):
        landmarker.detect_async(self.test_image, 0)

  def test_calling_detect_in_video_mode(self):
    options = _PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO,
    )
    with _PoseLandmarker.create_from_options(options) as landmarker:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the image mode'
      ):
        landmarker.detect(self.test_image)

  def test_calling_detect_async_in_video_mode(self):
    options = _PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO,
    )
    with _PoseLandmarker.create_from_options(options) as landmarker:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the live stream mode'
      ):
        landmarker.detect_async(self.test_image, 0)

  def test_detect_for_video_with_out_of_order_timestamp(self):
    options = _PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO,
    )
    with _PoseLandmarker.create_from_options(options) as landmarker:
      unused_result = landmarker.detect_for_video(self.test_image, 1)
      with self.assertRaisesRegex(
          ValueError, r'Input timestamp must be monotonically increasing'
      ):
        landmarker.detect_for_video(self.test_image, 0)

  @parameterized.parameters(
      (
          _POSE_IMAGE,
          0,
          False,
          _get_expected_pose_landmarker_result(_POSE_LANDMARKS),
      ),
      (
          _POSE_IMAGE,
          0,
          True,
          _get_expected_pose_landmarker_result(_POSE_LANDMARKS),
      ),
      (_BURGER_IMAGE, 0, False, PoseLandmarkerResult([], [])),
  )
  def test_detect_for_video(
      self, image_path, rotation, output_segmentation_masks, expected_result
  ):
    test_image = _Image.create_from_file(
        test_utils.get_test_data_path(image_path)
    )
    # Set rotation parameters using ImageProcessingOptions.
    image_processing_options = _ImageProcessingOptions(
        rotation_degrees=rotation
    )
    options = _PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        output_segmentation_masks=output_segmentation_masks,
        running_mode=_RUNNING_MODE.VIDEO,
    )
    with _PoseLandmarker.create_from_options(options) as landmarker:
      for timestamp in range(0, 300, 30):
        result = landmarker.detect_for_video(
            test_image, timestamp, image_processing_options
        )
        if result.pose_landmarks:
          self._expect_pose_landmarker_results_correct(
              result,
              expected_result,
              output_segmentation_masks,
              _LANDMARKS_MARGIN,
          )
        else:
          self.assertEqual(result, expected_result)

  def test_calling_detect_in_live_stream_mode(self):
    options = _PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=mock.MagicMock(),
    )
    with _PoseLandmarker.create_from_options(options) as landmarker:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the image mode'
      ):
        landmarker.detect(self.test_image)

  def test_calling_detect_for_video_in_live_stream_mode(self):
    options = _PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=mock.MagicMock(),
    )
    with _PoseLandmarker.create_from_options(options) as landmarker:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the video mode'
      ):
        landmarker.detect_for_video(self.test_image, 0)

  def test_detect_async_calls_with_illegal_timestamp(self):
    options = _PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=mock.MagicMock(),
    )
    with _PoseLandmarker.create_from_options(options) as landmarker:
      landmarker.detect_async(self.test_image, 100)
      with self.assertRaisesRegex(
          ValueError, r'Input timestamp must be monotonically increasing'
      ):
        landmarker.detect_async(self.test_image, 0)

  @parameterized.parameters(
      (
          _POSE_IMAGE,
          0,
          False,
          _get_expected_pose_landmarker_result(_POSE_LANDMARKS),
      ),
      (
          _POSE_IMAGE,
          0,
          True,
          _get_expected_pose_landmarker_result(_POSE_LANDMARKS),
      ),
      (_BURGER_IMAGE, 0, False, PoseLandmarkerResult([], [])),
  )
  def test_detect_async_calls(
      self, image_path, rotation, output_segmentation_masks, expected_result
  ):
    test_image = _Image.create_from_file(
        test_utils.get_test_data_path(image_path)
    )
    # Set rotation parameters using ImageProcessingOptions.
    image_processing_options = _ImageProcessingOptions(
        rotation_degrees=rotation
    )
    observed_timestamp_ms = -1

    def check_result(
        result: PoseLandmarkerResult, output_image: _Image, timestamp_ms: int
    ):
      if result.pose_landmarks:
        self._expect_pose_landmarker_results_correct(
            result,
            expected_result,
            output_segmentation_masks,
            _LANDMARKS_MARGIN,
        )
      else:
        self.assertEqual(result, expected_result)
      self.assertTrue(
          np.array_equal(output_image.numpy_view(), test_image.numpy_view())
      )
      self.assertLess(observed_timestamp_ms, timestamp_ms)
      self.observed_timestamp_ms = timestamp_ms

    options = _PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        output_segmentation_masks=output_segmentation_masks,
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=check_result,
    )
    with _PoseLandmarker.create_from_options(options) as landmarker:
      for timestamp in range(0, 300, 30):
        landmarker.detect_async(test_image, timestamp, image_processing_options)


if __name__ == '__main__':
  absltest.main()
