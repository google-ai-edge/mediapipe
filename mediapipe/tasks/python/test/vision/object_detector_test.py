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
"""Tests for object detector."""

import enum
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from mediapipe.python._framework_bindings import image as image_module
from mediapipe.tasks.python.components.containers import bounding_box as bounding_box_module
from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.components.containers import detections as detections_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.python.vision import object_detector
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

_BaseOptions = base_options_module.BaseOptions
_Category = category_module.Category
_BoundingBox = bounding_box_module.BoundingBox
_Detection = detections_module.Detection
_DetectionResult = detections_module.DetectionResult
_Image = image_module.Image
_ObjectDetector = object_detector.ObjectDetector
_ObjectDetectorOptions = object_detector.ObjectDetectorOptions
_RUNNING_MODE = running_mode_module.VisionTaskRunningMode

_MODEL_FILE = 'coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.tflite'
_IMAGE_FILE = 'cats_and_dogs.jpg'
_EXPECTED_DETECTION_RESULT = _DetectionResult(detections=[
    _Detection(
        bounding_box=_BoundingBox(
            origin_x=608, origin_y=161, width=381, height=439),
        categories=[
            _Category(
                index=None,
                score=0.69921875,
                display_name=None,
                category_name='cat')
        ]),
    _Detection(
        bounding_box=_BoundingBox(
            origin_x=60, origin_y=398, width=386, height=196),
        categories=[
            _Category(
                index=None,
                score=0.64453125,
                display_name=None,
                category_name='cat')
        ]),
    _Detection(
        bounding_box=_BoundingBox(
            origin_x=256, origin_y=395, width=173, height=202),
        categories=[
            _Category(
                index=None,
                score=0.51171875,
                display_name=None,
                category_name='cat')
        ]),
    _Detection(
        bounding_box=_BoundingBox(
            origin_x=362, origin_y=191, width=325, height=419),
        categories=[
            _Category(
                index=None,
                score=0.48828125,
                display_name=None,
                category_name='cat')
        ])
])
_ALLOW_LIST = ['cat', 'dog']
_DENY_LIST = ['cat']
_SCORE_THRESHOLD = 0.3
_MAX_RESULTS = 3
_TEST_DATA_DIR = 'mediapipe/tasks/testdata/vision'


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class ObjectDetectorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_image = _Image.create_from_file(
        test_utils.get_test_data_path(
            os.path.join(_TEST_DATA_DIR, _IMAGE_FILE)))
    self.model_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, _MODEL_FILE))

  def test_create_from_file_succeeds_with_valid_model_path(self):
    # Creates with default option and valid model file successfully.
    with _ObjectDetector.create_from_model_path(self.model_path) as detector:
      self.assertIsInstance(detector, _ObjectDetector)

  def test_create_from_options_succeeds_with_valid_model_path(self):
    # Creates with options containing model file successfully.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _ObjectDetectorOptions(base_options=base_options)
    with _ObjectDetector.create_from_options(options) as detector:
      self.assertIsInstance(detector, _ObjectDetector)

  def test_create_from_options_fails_with_invalid_model_path(self):
    with self.assertRaisesRegex(
        RuntimeError, 'Unable to open file at /path/to/invalid/model.tflite'):
      base_options = _BaseOptions(
          model_asset_path='/path/to/invalid/model.tflite')
      options = _ObjectDetectorOptions(base_options=base_options)
      _ObjectDetector.create_from_options(options)

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.model_path, 'rb') as f:
      base_options = _BaseOptions(model_asset_buffer=f.read())
      options = _ObjectDetectorOptions(base_options=base_options)
      detector = _ObjectDetector.create_from_options(options)
      self.assertIsInstance(detector, _ObjectDetector)

  @parameterized.parameters(
      (ModelFileType.FILE_NAME, 4, _EXPECTED_DETECTION_RESULT),
      (ModelFileType.FILE_CONTENT, 4, _EXPECTED_DETECTION_RESULT))
  def test_detect(self, model_file_type, max_results,
                  expected_detection_result):
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

    options = _ObjectDetectorOptions(
        base_options=base_options, max_results=max_results)
    detector = _ObjectDetector.create_from_options(options)

    # Performs object detection on the input.
    detection_result = detector.detect(self.test_image)
    # Comparing results.
    self.assertEqual(detection_result, expected_detection_result)
    # Closes the detector explicitly when the detector is not used in
    # a context.
    detector.close()

  @parameterized.parameters(
      (ModelFileType.FILE_NAME, 4, _EXPECTED_DETECTION_RESULT),
      (ModelFileType.FILE_CONTENT, 4, _EXPECTED_DETECTION_RESULT))
  def test_detect_in_context(self, model_file_type, max_results,
                             expected_detection_result):
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(model_asset_path=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, 'rb') as f:
        model_contents = f.read()
      base_options = _BaseOptions(model_asset_buffer=model_contents)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    options = _ObjectDetectorOptions(
        base_options=base_options, max_results=max_results)
    with _ObjectDetector.create_from_options(options) as detector:
      # Performs object detection on the input.
      detection_result = detector.detect(self.test_image)
      # Comparing results.
      self.assertEqual(detection_result, expected_detection_result)

  def test_score_threshold_option(self):
    options = _ObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        score_threshold=_SCORE_THRESHOLD)
    with _ObjectDetector.create_from_options(options) as detector:
      # Performs object detection on the input.
      detection_result = detector.detect(self.test_image)
      detections = detection_result.detections

      for detection in detections:
        score = detection.categories[0].score
        self.assertGreaterEqual(
            score, _SCORE_THRESHOLD,
            f'Detection with score lower than threshold found. {detection}')

  def test_max_results_option(self):
    options = _ObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        max_results=_MAX_RESULTS)
    with _ObjectDetector.create_from_options(options) as detector:
      # Performs object detection on the input.
      detection_result = detector.detect(self.test_image)
      detections = detection_result.detections

      self.assertLessEqual(
          len(detections), _MAX_RESULTS, 'Too many results returned.')

  def test_allow_list_option(self):
    options = _ObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        category_allowlist=_ALLOW_LIST)
    with _ObjectDetector.create_from_options(options) as detector:
      # Performs object detection on the input.
      detection_result = detector.detect(self.test_image)
      detections = detection_result.detections

      for detection in detections:
        label = detection.categories[0].category_name
        self.assertIn(label, _ALLOW_LIST,
                      f'Label {label} found but not in label allow list')

  def test_deny_list_option(self):
    options = _ObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        category_denylist=_DENY_LIST)
    with _ObjectDetector.create_from_options(options) as detector:
      # Performs object detection on the input.
      detection_result = detector.detect(self.test_image)
      detections = detection_result.detections

      for detection in detections:
        label = detection.categories[0].category_name
        self.assertNotIn(label, _DENY_LIST,
                         f'Label {label} found but in deny list.')

  def test_combined_allowlist_and_denylist(self):
    # Fails with combined allowlist and denylist
    with self.assertRaisesRegex(
        ValueError,
        r'`category_allowlist` and `category_denylist` are mutually '
        r'exclusive options.'):
      options = _ObjectDetectorOptions(
          base_options=_BaseOptions(model_asset_path=self.model_path),
          category_allowlist=['foo'],
          category_denylist=['bar'])
      with _ObjectDetector.create_from_options(options) as unused_detector:
        pass

  def test_empty_detection_outputs(self):
    options = _ObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        score_threshold=1)
    with _ObjectDetector.create_from_options(options) as detector:
      # Performs object detection on the input.
      detection_result = detector.detect(self.test_image)
      self.assertEmpty(detection_result.detections)

  def test_missing_result_callback(self):
    options = _ObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM)
    with self.assertRaisesRegex(ValueError,
                                r'result callback must be provided'):
      with _ObjectDetector.create_from_options(options) as unused_detector:
        pass

  @parameterized.parameters((_RUNNING_MODE.IMAGE), (_RUNNING_MODE.VIDEO))
  def test_illegal_result_callback(self, running_mode):
    options = _ObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=running_mode,
        result_callback=mock.MagicMock())
    with self.assertRaisesRegex(ValueError,
                                r'result callback should not be provided'):
      with _ObjectDetector.create_from_options(options) as unused_detector:
        pass

  def test_calling_detect_for_video_in_image_mode(self):
    options = _ObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.IMAGE)
    with _ObjectDetector.create_from_options(options) as detector:
      with self.assertRaisesRegex(ValueError,
                                  r'not initialized with the video mode'):
        detector.detect_for_video(self.test_image, 0)

  def test_calling_detect_async_in_image_mode(self):
    options = _ObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.IMAGE)
    with _ObjectDetector.create_from_options(options) as detector:
      with self.assertRaisesRegex(ValueError,
                                  r'not initialized with the live stream mode'):
        detector.detect_async(self.test_image, 0)

  def test_calling_detect_in_video_mode(self):
    options = _ObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO)
    with _ObjectDetector.create_from_options(options) as detector:
      with self.assertRaisesRegex(ValueError,
                                  r'not initialized with the image mode'):
        detector.detect(self.test_image)

  def test_calling_detect_async_in_video_mode(self):
    options = _ObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO)
    with _ObjectDetector.create_from_options(options) as detector:
      with self.assertRaisesRegex(ValueError,
                                  r'not initialized with the live stream mode'):
        detector.detect_async(self.test_image, 0)

  def test_detect_for_video_with_out_of_order_timestamp(self):
    options = _ObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO)
    with _ObjectDetector.create_from_options(options) as detector:
      unused_result = detector.detect_for_video(self.test_image, 1)
      with self.assertRaisesRegex(
          ValueError, r'Input timestamp must be monotonically increasing'):
        detector.detect_for_video(self.test_image, 0)

  # TODO: Tests how `detect_for_video` handles the temporal data
  # with a real video.
  def test_detect_for_video(self):
    options = _ObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.VIDEO,
        max_results=4)
    with _ObjectDetector.create_from_options(options) as detector:
      for timestamp in range(0, 300, 30):
        detection_result = detector.detect_for_video(self.test_image, timestamp)
        self.assertEqual(detection_result, _EXPECTED_DETECTION_RESULT)

  def test_calling_detect_in_live_stream_mode(self):
    options = _ObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=mock.MagicMock())
    with _ObjectDetector.create_from_options(options) as detector:
      with self.assertRaisesRegex(ValueError,
                                  r'not initialized with the image mode'):
        detector.detect(self.test_image)

  def test_calling_detect_for_video_in_live_stream_mode(self):
    options = _ObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        result_callback=mock.MagicMock())
    with _ObjectDetector.create_from_options(options) as detector:
      with self.assertRaisesRegex(ValueError,
                                  r'not initialized with the video mode'):
        detector.detect_for_video(self.test_image, 0)

  def test_detect_async_calls_with_illegal_timestamp(self):
    options = _ObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        max_results=4,
        result_callback=mock.MagicMock())
    with _ObjectDetector.create_from_options(options) as detector:
      detector.detect_async(self.test_image, 100)
      with self.assertRaisesRegex(
          ValueError, r'Input timestamp must be monotonically increasing'):
        detector.detect_async(self.test_image, 0)

  @parameterized.parameters((0, _EXPECTED_DETECTION_RESULT),
                            (1, _DetectionResult(detections=[])))
  def test_detect_async_calls(self, threshold, expected_result):
    observed_timestamp_ms = -1

    def check_result(result: _DetectionResult, output_image: _Image,
                     timestamp_ms: int):
      self.assertEqual(result, expected_result)
      self.assertTrue(
          np.array_equal(output_image.numpy_view(),
                         self.test_image.numpy_view()))
      self.assertLess(observed_timestamp_ms, timestamp_ms)
      self.observed_timestamp_ms = timestamp_ms

    options = _ObjectDetectorOptions(
        base_options=_BaseOptions(model_asset_path=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        max_results=4,
        score_threshold=threshold,
        result_callback=check_result)
    detector = _ObjectDetector.create_from_options(options)
    for timestamp in range(0, 300, 30):
      detector.detect_async(self.test_image, timestamp)
    detector.close()


if __name__ == '__main__':
  absltest.main()
