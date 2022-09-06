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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from mediapipe.python._framework_bindings import image as image_module
from mediapipe.tasks.python.components.containers import bounding_box as bounding_box_module
from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.components.containers import detections as detections_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_util
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
            origin_x=608, origin_y=164, width=381, height=432),
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
            origin_x=257, origin_y=394, width=173, height=202),
        categories=[
            _Category(
                index=None,
                score=0.51171875,
                display_name=None,
                category_name='cat')
        ]),
    _Detection(
        bounding_box=_BoundingBox(
            origin_x=362, origin_y=195, width=325, height=412),
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


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class ObjectDetectorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_image = test_util.read_test_image(
        test_util.get_test_data_path(_IMAGE_FILE))
    self.model_path = test_util.get_test_data_path(_MODEL_FILE)

  def test_create_from_file_succeeds_with_valid_model_path(self):
    # Creates with default option and valid model file successfully.
    with _ObjectDetector.create_from_model_path(self.model_path) as detector:
      self.assertIsInstance(detector, _ObjectDetector)

  def test_create_from_options_succeeds_with_valid_model_path(self):
    # Creates with options containing model file successfully.
    base_options = _BaseOptions(file_name=self.model_path)
    options = _ObjectDetectorOptions(base_options=base_options)
    with _ObjectDetector.create_from_options(options) as detector:
      self.assertIsInstance(detector, _ObjectDetector)

  def test_create_from_options_fails_with_invalid_model_path(self):
    # Invalid empty model path.
    with self.assertRaisesRegex(
        ValueError,
        r"ExternalFile must specify at least one of 'file_content', "
        r"'file_name' or 'file_descriptor_meta'."):
      base_options = _BaseOptions(file_name='')
      options = _ObjectDetectorOptions(base_options=base_options)
      _ObjectDetector.create_from_options(options)

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.model_path, 'rb') as f:
      base_options = _BaseOptions(file_content=f.read())
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
      base_options = _BaseOptions(file_name=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(file_content=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    options = _ObjectDetectorOptions(
        base_options=base_options, max_results=max_results)
    detector = _ObjectDetector.create_from_options(options)

    # Performs object detection on the input.
    image_result = detector.detect(self.test_image)
    # Comparing results.
    self.assertEqual(image_result, expected_detection_result)
    # Closes the detector explicitly when the detector is not used in
    # a context.
    detector.close()

  @parameterized.parameters(
      (ModelFileType.FILE_NAME, 4, _EXPECTED_DETECTION_RESULT),
      (ModelFileType.FILE_CONTENT, 4, _EXPECTED_DETECTION_RESULT))
  def test_detect_in_context(self, model_file_type, max_results,
                             expected_detection_result):
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(file_name=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(file_content=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    options = _ObjectDetectorOptions(
        base_options=base_options, max_results=max_results)
    with _ObjectDetector.create_from_options(options) as detector:
      # Performs object detection on the input.
      image_result = detector.detect(self.test_image)
      # Comparing results.
      self.assertEqual(image_result, expected_detection_result)

  def test_score_threshold_option(self):
    options = _ObjectDetectorOptions(
        base_options=_BaseOptions(file_name=self.model_path),
        score_threshold=_SCORE_THRESHOLD)
    with _ObjectDetector.create_from_options(options) as detector:
      # Performs object detection on the input.
      image_result = detector.detect(self.test_image)
      detections = image_result.detections

      for detection in detections:
        score = detection.categories[0].score
        self.assertGreaterEqual(
            score, _SCORE_THRESHOLD,
            f'Detection with score lower than threshold found. {detection}')

  def test_max_results_option(self):
    options = _ObjectDetectorOptions(
        base_options=_BaseOptions(file_name=self.model_path),
        max_results=_MAX_RESULTS)
    with _ObjectDetector.create_from_options(options) as detector:
      # Performs object detection on the input.
      image_result = detector.detect(self.test_image)
      detections = image_result.detections

      self.assertLessEqual(
          len(detections), _MAX_RESULTS, 'Too many results returned.')

  def test_allow_list_option(self):
    options = _ObjectDetectorOptions(
        base_options=_BaseOptions(file_name=self.model_path),
        category_allowlist=_ALLOW_LIST)
    with _ObjectDetector.create_from_options(options) as detector:
      # Performs object detection on the input.
      image_result = detector.detect(self.test_image)
      detections = image_result.detections

      for detection in detections:
        label = detection.categories[0].category_name
        self.assertIn(label, _ALLOW_LIST,
                      f'Label {label} found but not in label allow list')

  def test_deny_list_option(self):
    options = _ObjectDetectorOptions(
        base_options=_BaseOptions(file_name=self.model_path),
        category_denylist=_DENY_LIST)
    with _ObjectDetector.create_from_options(options) as detector:
      # Performs object detection on the input.
      image_result = detector.detect(self.test_image)
      detections = image_result.detections

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
          base_options=_BaseOptions(file_name=self.model_path),
          category_allowlist=['foo'],
          category_denylist=['bar'])
      with _ObjectDetector.create_from_options(options) as unused_detector:
        pass

  def test_empty_detection_outputs(self):
    options = _ObjectDetectorOptions(
        base_options=_BaseOptions(file_name=self.model_path), score_threshold=1)
    with _ObjectDetector.create_from_options(options) as detector:
      # Performs object detection on the input.
      image_result = detector.detect(self.test_image)
      self.assertEmpty(image_result.detections)

  def test_missing_result_callback(self):
    options = _ObjectDetectorOptions(
        base_options=_BaseOptions(file_name=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM)
    with self.assertRaisesRegex(ValueError,
                                r'result callback must be provided'):
      with _ObjectDetector.create_from_options(options) as unused_detector:
        pass

  @parameterized.parameters((_RUNNING_MODE.IMAGE), (_RUNNING_MODE.VIDEO))
  def test_illegal_result_callback(self, running_mode):

    def pass_through(unused_result: _DetectionResult,
                     unused_output_image: _Image, unused_timestamp_ms: int):
      pass

    options = _ObjectDetectorOptions(
        base_options=_BaseOptions(file_name=self.model_path),
        running_mode=running_mode,
        result_callback=pass_through)
    with self.assertRaisesRegex(ValueError,
                                r'result callback should not be provided'):
      with _ObjectDetector.create_from_options(options) as unused_detector:
        pass

  def test_detect_async_calls_with_illegal_timestamp(self):

    def pass_through(unused_result: _DetectionResult,
                     unused_output_image: _Image, unused_timestamp_ms: int):
      pass

    options = _ObjectDetectorOptions(
        base_options=_BaseOptions(file_name=self.model_path),
        running_mode=_RUNNING_MODE.LIVE_STREAM,
        max_results=4,
        result_callback=pass_through)
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
        base_options=_BaseOptions(file_name=self.model_path),
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
