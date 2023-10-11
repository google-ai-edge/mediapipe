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
"""Tests for face detector."""

import enum
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from google.protobuf import text_format
from mediapipe.framework.formats import detection_pb2
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.tasks.python.components.containers import bounding_box as bounding_box_module
from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.components.containers import detections as detections_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.python.vision import face_detector
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

FaceDetectorResult = detections_module.DetectionResult
_BaseOptions = base_options_module.BaseOptions
_Category = category_module.Category
_BoundingBox = bounding_box_module.BoundingBox
_Detection = detections_module.Detection
_Image = image_module.Image
_FaceDetector = face_detector.FaceDetector
_FaceDetectorOptions = face_detector.FaceDetectorOptions
_RUNNING_MODE = running_mode_module.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions

_SHORT_RANGE_BLAZE_FACE_MODEL = 'face_detection_short_range.tflite'
_PORTRAIT_IMAGE = 'portrait.jpg'
_PORTRAIT_EXPECTED_DETECTION = 'portrait_expected_detection.pbtxt'
_PORTRAIT_ROTATED_IMAGE = 'portrait_rotated.jpg'
_PORTRAIT_ROTATED_EXPECTED_DETECTION = (
    'portrait_rotated_expected_detection.pbtxt'
)
_CAT_IMAGE = 'cat.jpg'
_KEYPOINT_ERROR_THRESHOLD = 1e-2
_TEST_DATA_DIR = 'mediapipe/tasks/testdata/vision'


def _get_expected_face_detector_result(file_name: str) -> FaceDetectorResult:
  face_detection_result_file_path = test_utils.get_test_data_path(
      os.path.join(_TEST_DATA_DIR, file_name)
  )
  with open(face_detection_result_file_path, 'rb') as f:
    face_detection_proto = detection_pb2.Detection()
    text_format.Parse(f.read(), face_detection_proto)
  face_detection = detections_module.Detection.create_from_pb2(
      face_detection_proto
  )
  return FaceDetectorResult(detections=[face_detection])


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class FaceDetectorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_image = _Image.create_from_file(
        test_utils.get_test_data_path(
            os.path.join(_TEST_DATA_DIR, _PORTRAIT_IMAGE)
        )
    )
    self.model_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, _SHORT_RANGE_BLAZE_FACE_MODEL)
    )

  def test_create_from_file_succeeds_with_valid_model_path(self):
    # Creates with default option and valid model file successfully.
    with _FaceDetector.create_from_model_path(self.model_path) as detector:
      self.assertIsInstance(detector, _FaceDetector)

  def test_create_from_options_succeeds_with_valid_model_path(self):
    # Creates with options containing model file successfully.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _FaceDetectorOptions(base_options=base_options)
    with _FaceDetector.create_from_options(options) as detector:
      self.assertIsInstance(detector, _FaceDetector)

  def test_create_from_options_fails_with_invalid_model_path(self):
    with self.assertRaisesRegex(
        RuntimeError, 'Unable to open file at /path/to/invalid/model.tflite'
    ):
      base_options = _BaseOptions(
          model_asset_path='/path/to/invalid/model.tflite'
      )
      options = _FaceDetectorOptions(base_options=base_options)
      _FaceDetector.create_from_options(options)

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.model_path, 'rb') as f:
      base_options = _BaseOptions(model_asset_buffer=f.read())
      options = _FaceDetectorOptions(base_options=base_options)
      detector = _FaceDetector.create_from_options(options)
      self.assertIsInstance(detector, _FaceDetector)

  def _expect_keypoints_correct(self, actual_keypoints, expected_keypoints):
    self.assertLen(actual_keypoints, len(expected_keypoints))
    for i in range(len(actual_keypoints)):
      self.assertAlmostEqual(
          actual_keypoints[i].x,
          expected_keypoints[i].x,
          delta=_KEYPOINT_ERROR_THRESHOLD,
      )
      self.assertAlmostEqual(
          actual_keypoints[i].y,
          expected_keypoints[i].y,
          delta=_KEYPOINT_ERROR_THRESHOLD,
      )

  def _expect_face_detector_results_correct(
      self, actual_results, expected_results
  ):
    self.assertLen(actual_results.detections, len(expected_results.detections))
    for i in range(len(actual_results.detections)):
      actual_bbox = actual_results.detections[i].bounding_box
      expected_bbox = expected_results.detections[i].bounding_box
      self.assertEqual(actual_bbox, expected_bbox)
      self.assertNotEmpty(actual_results.detections[i].keypoints)
      self._expect_keypoints_correct(
          actual_results.detections[i].keypoints,
          expected_results.detections[i].keypoints,
      )

  @parameterized.parameters(
      (ModelFileType.FILE_NAME, _PORTRAIT_EXPECTED_DETECTION),
      (ModelFileType.FILE_CONTENT, _PORTRAIT_EXPECTED_DETECTION),
  )
  def test_detect(self, model_file_type, expected_detection_result_file):
    # Creates detector.
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(model_asset_path=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(model_asset_buffer=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    options = _FaceDetectorOptions(base_options=base_options)
    detector = _FaceDetector.create_from_options(options)

    # Performs face detection on the input.
    detection_result = detector.detect(self.test_image)
    # Comparing results.
    expected_detection_result = _get_expected_face_detector_result(
        expected_detection_result_file
    )
    self._expect_face_detector_results_correct(
        detection_result, expected_detection_result
    )
    # Closes the detector explicitly when the detector is not used in
    # a context.
    detector.close()

  @parameterized.parameters(
      (ModelFileType.FILE_NAME, _PORTRAIT_EXPECTED_DETECTION),
      (ModelFileType.FILE_CONTENT, _PORTRAIT_EXPECTED_DETECTION),
  )
  def test_detect_in_context(
      self, model_file_type, expected_detection_result_file
  ):
    # Creates detector.
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(model_asset_path=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(model_asset_buffer=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    options = _FaceDetectorOptions(base_options=base_options)

    with _FaceDetector.create_from_options(options) as detector:
      # Performs face detection on the input.
      detection_result = detector.detect(self.test_image)
      # Comparing results.
      expected_detection_result = _get_expected_face_detector_result(
          expected_detection_result_file
      )
      self._expect_face_detector_results_correct(
          detection_result, expected_detection_result
      )

  def test_detect_succeeds_with_rotated_image(self):
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _FaceDetectorOptions(base_options=base_options)
    with _FaceDetector.create_from_options(options) as detector:
      # Load the test image.
      test_image = _Image.create_from_file(
          test_utils.get_test_data_path(
              os.path.join(_TEST_DATA_DIR, _PORTRAIT_ROTATED_IMAGE)
          )
      )
      # Rotated input image.
      image_processing_options = _ImageProcessingOptions(rotation_degrees=-90)
      # Performs face detection on the input.
      detection_result = detector.detect(test_image, image_processing_options)
      # Comparing results.
      expected_detection_result = _get_expected_face_detector_result(
          _PORTRAIT_ROTATED_EXPECTED_DETECTION
      )
      self._expect_face_detector_results_correct(
          detection_result, expected_detection_result
      )

  def test_empty_detection_outputs(self):
    # Load a test image with no faces.
    test_image = _Image.create_from_file(
        test_utils.get_test_data_path(os.path.join(_TEST_DATA_DIR, _CAT_IMAGE))
    )
    options = _FaceDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path)
    )
    with _FaceDetector.create_from_options(options) as detector:
      # Performs face detection on the input.
      detection_result = detector.detect(test_image)
      self.assertEmpty(detection_result.detections)

  def test_missing_result_callback(self):
    options = _FaceDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
    )
    with self.assertRaisesRegex(
        ValueError, r'result callback must be provided'
    ):
      with _FaceDetector.create_from_options(options) as unused_detector:
        pass

  @parameterized.parameters((_RUNNING_MODE.IMAGE), (_RUNNING_MODE.VIDEO))
  def test_illegal_result_callback(self, running_mode):
    options = _FaceDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=running_mode,
        result_callback=mock.MagicMock(),
    )
    with self.assertRaisesRegex(
        ValueError, r'result callback should not be provided'
    ):
      with _FaceDetector.create_from_options(options) as unused_detector:
        pass

  def test_calling_detect_for_video_in_image_mode(self):
    options = _FaceDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.IMAGE,
    )
    with _FaceDetector.create_from_options(options) as detector:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the video mode'
      ):
        detector.detect_for_video(self.test_image, 0)

  def test_calling_detect_async_in_image_mode(self):
    options = _FaceDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.IMAGE,
    )
    with _FaceDetector.create_from_options(options) as detector:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the live stream mode'
      ):
        detector.detect_async(self.test_image, 0)

  def test_calling_detect_in_video_mode(self):
    options = _FaceDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO,
    )
    with _FaceDetector.create_from_options(options) as detector:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the image mode'
      ):
        detector.detect(self.test_image)

  def test_calling_detect_async_in_video_mode(self):
    options = _FaceDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO,
    )
    with _FaceDetector.create_from_options(options) as detector:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the live stream mode'
      ):
        detector.detect_async(self.test_image, 0)

  def test_detect_for_video_with_out_of_order_timestamp(self):
    options = _FaceDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO,
    )
    with _FaceDetector.create_from_options(options) as detector:
      unused_result = detector.detect_for_video(self.test_image, 1)
      with self.assertRaisesRegex(
          ValueError, r'Input timestamp must be monotonically increasing'
      ):
        detector.detect_for_video(self.test_image, 0)

  @parameterized.parameters(
      (
          ModelFileType.FILE_NAME,
          _PORTRAIT_IMAGE,
          0,
          _get_expected_face_detector_result(_PORTRAIT_EXPECTED_DETECTION),
      ),
      (
          ModelFileType.FILE_CONTENT,
          _PORTRAIT_IMAGE,
          0,
          _get_expected_face_detector_result(_PORTRAIT_EXPECTED_DETECTION),
      ),
      (
          ModelFileType.FILE_NAME,
          _PORTRAIT_ROTATED_IMAGE,
          -90,
          _get_expected_face_detector_result(
              _PORTRAIT_ROTATED_EXPECTED_DETECTION
          ),
      ),
      (
          ModelFileType.FILE_CONTENT,
          _PORTRAIT_ROTATED_IMAGE,
          -90,
          _get_expected_face_detector_result(
              _PORTRAIT_ROTATED_EXPECTED_DETECTION
          ),
      ),
      (ModelFileType.FILE_NAME, _CAT_IMAGE, 0, FaceDetectorResult([])),
      (ModelFileType.FILE_CONTENT, _CAT_IMAGE, 0, FaceDetectorResult([])),
  )
  def test_detect_for_video(
      self,
      model_file_type,
      test_image_file_name,
      rotation_degrees,
      expected_detection_result,
  ):
    # Creates detector.
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(model_asset_path=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(model_asset_buffer=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    options = _FaceDetectorOptions(
        base_options=base_options, running_mode=_RUNNING_MODE.VIDEO
    )

    with _FaceDetector.create_from_options(options) as detector:
      for timestamp in range(0, 300, 30):
        # Load the test image.
        test_image = _Image.create_from_file(
            test_utils.get_test_data_path(
                os.path.join(_TEST_DATA_DIR, test_image_file_name)
            )
        )
        # Set the image processing options.
        image_processing_options = _ImageProcessingOptions(
            rotation_degrees=rotation_degrees
        )
        # Performs face detection on the input.
        detection_result = detector.detect_for_video(
            test_image, timestamp, image_processing_options
        )
        # Comparing results.
        self._expect_face_detector_results_correct(
            detection_result, expected_detection_result
        )

  def test_calling_detect_in_live_stream_mode(self):
    options = _FaceDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=mock.MagicMock(),
    )
    with _FaceDetector.create_from_options(options) as detector:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the image mode'
      ):
        detector.detect(self.test_image)

  def test_calling_detect_for_video_in_live_stream_mode(self):
    options = _FaceDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=mock.MagicMock(),
    )
    with _FaceDetector.create_from_options(options) as detector:
      with self.assertRaisesRegex(
          ValueError, r'not initialized with the video mode'
      ):
        detector.detect_for_video(self.test_image, 0)

  def test_detect_async_calls_with_illegal_timestamp(self):
    options = _FaceDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=mock.MagicMock(),
    )
    with _FaceDetector.create_from_options(options) as detector:
      detector.detect_async(self.test_image, 100)
      with self.assertRaisesRegex(
          ValueError, r'Input timestamp must be monotonically increasing'
      ):
        detector.detect_async(self.test_image, 0)

  @parameterized.parameters(
      (
          ModelFileType.FILE_NAME,
          _PORTRAIT_IMAGE,
          0,
          _get_expected_face_detector_result(_PORTRAIT_EXPECTED_DETECTION),
      ),
      (
          ModelFileType.FILE_CONTENT,
          _PORTRAIT_IMAGE,
          0,
          _get_expected_face_detector_result(_PORTRAIT_EXPECTED_DETECTION),
      ),
      (
          ModelFileType.FILE_NAME,
          _PORTRAIT_ROTATED_IMAGE,
          -90,
          _get_expected_face_detector_result(
              _PORTRAIT_ROTATED_EXPECTED_DETECTION
          ),
      ),
      (
          ModelFileType.FILE_CONTENT,
          _PORTRAIT_ROTATED_IMAGE,
          -90,
          _get_expected_face_detector_result(
              _PORTRAIT_ROTATED_EXPECTED_DETECTION
          ),
      ),
      (ModelFileType.FILE_NAME, _CAT_IMAGE, 0, FaceDetectorResult([])),
      (ModelFileType.FILE_CONTENT, _CAT_IMAGE, 0, FaceDetectorResult([])),
  )
  def test_detect_async_calls(
      self,
      model_file_type,
      test_image_file_name,
      rotation_degrees,
      expected_detection_result,
  ):
    # Creates detector.
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(model_asset_path=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(model_asset_buffer=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    observed_timestamp_ms = -1

    def check_result(
        result: FaceDetectorResult,
        unused_output_image: _Image,
        timestamp_ms: int,
    ):
      self._expect_face_detector_results_correct(
          result, expected_detection_result
      )
      self.assertLess(observed_timestamp_ms, timestamp_ms)
      self.observed_timestamp_ms = timestamp_ms

    options = _FaceDetectorOptions(
        base_options=base_options,
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=check_result,
    )

    # Load the test image.
    test_image = _Image.create_from_file(
        test_utils.get_test_data_path(
            os.path.join(_TEST_DATA_DIR, test_image_file_name)
        )
    )

    with _FaceDetector.create_from_options(options) as detector:
      for timestamp in range(0, 300, 30):
        # Set the image processing options.
        image_processing_options = _ImageProcessingOptions(
            rotation_degrees=rotation_degrees
        )
        detector.detect_async(test_image, timestamp, image_processing_options)


if __name__ == '__main__':
  absltest.main()
