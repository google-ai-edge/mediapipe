# Copyright 2020 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for mediapipe.python.solutions.drawing_utils."""

from absl.testing import absltest
from absl.testing import parameterized
import cv2
import numpy as np

from google.protobuf import text_format

from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import drawing_utils

DEFAULT_BBOX_DRAWING_SPEC = drawing_utils.DrawingSpec()
DEFAULT_CONNECTION_DRAWING_SPEC = drawing_utils.DrawingSpec()
DEFAULT_CIRCLE_DRAWING_SPEC = drawing_utils.DrawingSpec(color=(0, 0, 255))
DEFAULT_AXIS_DRAWING_SPEC = drawing_utils.DrawingSpec()
DEFAULT_CYCLE_BORDER_COLOR = (224, 224, 224)


class DrawingUtilTest(parameterized.TestCase):

  def test_invalid_input_image(self):
    image = np.arange(18, dtype=np.uint8).reshape(3, 3, 2)
    with self.assertRaisesRegex(
        ValueError, 'Input image must contain three channel rgb data.'):
      drawing_utils.draw_landmarks(image, landmark_pb2.NormalizedLandmarkList())
    with self.assertRaisesRegex(
        ValueError, 'Input image must contain three channel rgb data.'):
      drawing_utils.draw_detection(image, detection_pb2.Detection())
    with self.assertRaisesRegex(
        ValueError, 'Input image must contain three channel rgb data.'):
      rotation = np.eye(3, dtype=np.float32)
      translation = np.array([0., 0., 1.])
      drawing_utils.draw_axis(image, rotation, translation)

  def test_invalid_connection(self):
    landmark_list = text_format.Parse(
        'landmark {x: 0.5 y: 0.5} landmark {x: 0.2 y: 0.2}',
        landmark_pb2.NormalizedLandmarkList())
    image = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    with self.assertRaisesRegex(ValueError, 'Landmark index is out of range.'):
      drawing_utils.draw_landmarks(image, landmark_list, [(0, 2)])

  def test_unqualified_detection(self):
    detection = text_format.Parse('location_data {format: GLOBAL}',
                                  detection_pb2.Detection())
    image = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    with self.assertRaisesRegex(ValueError, 'LocationData must be relative'):
      drawing_utils.draw_detection(image, detection)

  def test_draw_keypoints_only(self):
    detection = text_format.Parse(
        'location_data {'
        '  format: RELATIVE_BOUNDING_BOX'
        '  relative_keypoints {x: 0 y: 1}'
        '  relative_keypoints {x: 1 y: 0}}', detection_pb2.Detection())
    image = np.zeros((100, 100, 3), np.uint8)
    expected_result = np.copy(image)
    cv2.circle(expected_result, (0, 99),
               DEFAULT_CIRCLE_DRAWING_SPEC.circle_radius,
               DEFAULT_CIRCLE_DRAWING_SPEC.color,
               DEFAULT_CIRCLE_DRAWING_SPEC.thickness)
    cv2.circle(expected_result, (99, 0),
               DEFAULT_CIRCLE_DRAWING_SPEC.circle_radius,
               DEFAULT_CIRCLE_DRAWING_SPEC.color,
               DEFAULT_CIRCLE_DRAWING_SPEC.thickness)
    drawing_utils.draw_detection(image, detection)
    np.testing.assert_array_equal(image, expected_result)

  def test_draw_bboxs_only(self):
    detection = text_format.Parse(
        'location_data {'
        '  format: RELATIVE_BOUNDING_BOX'
        '  relative_bounding_box {xmin: 0 ymin: 0 width: 1 height: 1}}',
        detection_pb2.Detection())
    image = np.zeros((100, 100, 3), np.uint8)
    expected_result = np.copy(image)
    cv2.rectangle(expected_result, (0, 0), (99, 99),
                  DEFAULT_BBOX_DRAWING_SPEC.color,
                  DEFAULT_BBOX_DRAWING_SPEC.thickness)
    drawing_utils.draw_detection(image, detection)
    np.testing.assert_array_equal(image, expected_result)

  @parameterized.named_parameters(
      ('landmark_list_has_only_one_element', 'landmark {x: 0.1 y: 0.1}'),
      ('second_landmark_is_invisible',
       'landmark {x: 0.1 y: 0.1} landmark {x: 0.5 y: 0.5 visibility: 0.0}'))
  def test_draw_single_landmark_point(self, landmark_list_text):
    landmark_list = text_format.Parse(landmark_list_text,
                                      landmark_pb2.NormalizedLandmarkList())
    image = np.zeros((100, 100, 3), np.uint8)
    expected_result = np.copy(image)
    cv2.circle(expected_result, (10, 10),
               DEFAULT_CIRCLE_DRAWING_SPEC.circle_radius + 1,
               DEFAULT_CYCLE_BORDER_COLOR,
               DEFAULT_CIRCLE_DRAWING_SPEC.thickness)
    cv2.circle(expected_result, (10, 10),
               DEFAULT_CIRCLE_DRAWING_SPEC.circle_radius,
               DEFAULT_CIRCLE_DRAWING_SPEC.color,
               DEFAULT_CIRCLE_DRAWING_SPEC.thickness)
    drawing_utils.draw_landmarks(image, landmark_list)
    np.testing.assert_array_equal(image, expected_result)

  @parameterized.named_parameters(
      ('landmarks_have_x_and_y_only',
       'landmark {x: 0.1 y: 0.5} landmark {x: 0.5 y: 0.1}'),
      ('landmark_zero_visibility_and_presence',
       'landmark {x: 0.1 y: 0.5 presence: 0.5}'
       'landmark {x: 0.5 y: 0.1 visibility: 0.5}'))
  def test_draw_landmarks_and_connections(self, landmark_list_text):
    landmark_list = text_format.Parse(landmark_list_text,
                                      landmark_pb2.NormalizedLandmarkList())
    image = np.zeros((100, 100, 3), np.uint8)
    expected_result = np.copy(image)
    start_point = (10, 50)
    end_point = (50, 10)
    cv2.line(expected_result, start_point, end_point,
             DEFAULT_CONNECTION_DRAWING_SPEC.color,
             DEFAULT_CONNECTION_DRAWING_SPEC.thickness)
    cv2.circle(expected_result, start_point,
               DEFAULT_CIRCLE_DRAWING_SPEC.circle_radius + 1,
               DEFAULT_CYCLE_BORDER_COLOR,
               DEFAULT_CIRCLE_DRAWING_SPEC.thickness)
    cv2.circle(expected_result, end_point,
               DEFAULT_CIRCLE_DRAWING_SPEC.circle_radius + 1,
               DEFAULT_CYCLE_BORDER_COLOR,
               DEFAULT_CIRCLE_DRAWING_SPEC.thickness)
    cv2.circle(expected_result, start_point,
               DEFAULT_CIRCLE_DRAWING_SPEC.circle_radius,
               DEFAULT_CIRCLE_DRAWING_SPEC.color,
               DEFAULT_CIRCLE_DRAWING_SPEC.thickness)
    cv2.circle(expected_result, end_point,
               DEFAULT_CIRCLE_DRAWING_SPEC.circle_radius,
               DEFAULT_CIRCLE_DRAWING_SPEC.color,
               DEFAULT_CIRCLE_DRAWING_SPEC.thickness)
    drawing_utils.draw_landmarks(
        image=image, landmark_list=landmark_list, connections=[(0, 1)])
    np.testing.assert_array_equal(image, expected_result)

  def test_draw_axis(self):
    image = np.zeros((100, 100, 3), np.uint8)
    expected_result = np.copy(image)
    origin = (50, 50)
    x_axis = (75, 50)
    y_axis = (50, 22)
    z_axis = (50, 77)
    cv2.arrowedLine(expected_result, origin, x_axis, drawing_utils.RED_COLOR,
                    DEFAULT_AXIS_DRAWING_SPEC.thickness)
    cv2.arrowedLine(expected_result, origin, y_axis, drawing_utils.GREEN_COLOR,
                    DEFAULT_AXIS_DRAWING_SPEC.thickness)
    cv2.arrowedLine(expected_result, origin, z_axis, drawing_utils.BLUE_COLOR,
                    DEFAULT_AXIS_DRAWING_SPEC.thickness)
    r = np.sqrt(2.) / 2.
    rotation = np.array([[1., 0., 0.], [0., r, -r], [0., r, r]])
    translation = np.array([0, 0, -0.2])
    drawing_utils.draw_axis(image, rotation, translation)
    np.testing.assert_array_equal(image, expected_result)

  def test_draw_axis_zero_translation(self):
    image = np.zeros((100, 100, 3), np.uint8)
    expected_result = np.copy(image)
    origin = (50, 50)
    x_axis = (0, 50)
    y_axis = (50, 100)
    z_axis = (50, 50)
    cv2.arrowedLine(expected_result, origin, x_axis, drawing_utils.RED_COLOR,
                    DEFAULT_AXIS_DRAWING_SPEC.thickness)
    cv2.arrowedLine(expected_result, origin, y_axis, drawing_utils.GREEN_COLOR,
                    DEFAULT_AXIS_DRAWING_SPEC.thickness)
    cv2.arrowedLine(expected_result, origin, z_axis, drawing_utils.BLUE_COLOR,
                    DEFAULT_AXIS_DRAWING_SPEC.thickness)
    rotation = np.eye(3, dtype=np.float32)
    translation = np.zeros((3,), dtype=np.float32)
    drawing_utils.draw_axis(image, rotation, translation)
    np.testing.assert_array_equal(image, expected_result)

  def test_min_and_max_coordinate_values(self):
    landmark_list = text_format.Parse(
        'landmark {x: 0.0 y: 1.0}'
        'landmark {x: 1.0 y: 0.0}', landmark_pb2.NormalizedLandmarkList())
    image = np.zeros((100, 100, 3), np.uint8)
    expected_result = np.copy(image)
    start_point = (0, 99)
    end_point = (99, 0)
    cv2.line(expected_result, start_point, end_point,
             DEFAULT_CONNECTION_DRAWING_SPEC.color,
             DEFAULT_CONNECTION_DRAWING_SPEC.thickness)
    cv2.circle(expected_result, start_point,
               DEFAULT_CIRCLE_DRAWING_SPEC.circle_radius + 1,
               DEFAULT_CYCLE_BORDER_COLOR,
               DEFAULT_CIRCLE_DRAWING_SPEC.thickness)
    cv2.circle(expected_result, end_point,
               DEFAULT_CIRCLE_DRAWING_SPEC.circle_radius + 1,
               DEFAULT_CYCLE_BORDER_COLOR,
               DEFAULT_CIRCLE_DRAWING_SPEC.thickness)
    cv2.circle(expected_result, start_point,
               DEFAULT_CIRCLE_DRAWING_SPEC.circle_radius,
               DEFAULT_CIRCLE_DRAWING_SPEC.color,
               DEFAULT_CIRCLE_DRAWING_SPEC.thickness)
    cv2.circle(expected_result, end_point,
               DEFAULT_CIRCLE_DRAWING_SPEC.circle_radius,
               DEFAULT_CIRCLE_DRAWING_SPEC.color,
               DEFAULT_CIRCLE_DRAWING_SPEC.thickness)
    drawing_utils.draw_landmarks(
        image=image, landmark_list=landmark_list, connections=[(0, 1)])
    np.testing.assert_array_equal(image, expected_result)

  def test_drawing_spec(self):
    landmark_list = text_format.Parse(
        'landmark {x: 0.1 y: 0.1}'
        'landmark {x: 0.8 y: 0.8}', landmark_pb2.NormalizedLandmarkList())
    image = np.zeros((100, 100, 3), np.uint8)
    landmark_drawing_spec = drawing_utils.DrawingSpec(
        color=(0, 0, 255), thickness=5)
    connection_drawing_spec = drawing_utils.DrawingSpec(
        color=(255, 0, 0), thickness=3)
    expected_result = np.copy(image)
    start_point = (10, 10)
    end_point = (80, 80)
    cv2.line(expected_result, start_point, end_point,
             connection_drawing_spec.color, connection_drawing_spec.thickness)
    cv2.circle(expected_result, start_point,
               landmark_drawing_spec.circle_radius + 1,
               DEFAULT_CYCLE_BORDER_COLOR, landmark_drawing_spec.thickness)
    cv2.circle(expected_result, end_point,
               landmark_drawing_spec.circle_radius + 1,
               DEFAULT_CYCLE_BORDER_COLOR, landmark_drawing_spec.thickness)
    cv2.circle(expected_result, start_point,
               landmark_drawing_spec.circle_radius, landmark_drawing_spec.color,
               landmark_drawing_spec.thickness)
    cv2.circle(expected_result, end_point, landmark_drawing_spec.circle_radius,
               landmark_drawing_spec.color, landmark_drawing_spec.thickness)
    drawing_utils.draw_landmarks(
        image=image,
        landmark_list=landmark_list,
        connections=[(0, 1)],
        landmark_drawing_spec=landmark_drawing_spec,
        connection_drawing_spec=connection_drawing_spec)
    np.testing.assert_array_equal(image, expected_result)


if __name__ == '__main__':
  absltest.main()
