# Copyright 2026 The MediaPipe Authors.
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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from mediapipe.tasks.python.components.containers import bounding_box as bounding_box_module
from mediapipe.tasks.python.components.containers import detections as detections_module
from mediapipe.tasks.python.components.containers import keypoint as keypoint_module
from mediapipe.tasks.python.components.containers import landmark as landmark_module
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import face_landmarker

_Connection = face_landmarker.FaceLandmarksConnections.Connection


class DrawingUtilsTest(parameterized.TestCase):

  @parameterized.parameters(3, 4)
  def test_draw_landmarks_supports_3_and_4_channel_images(self, channels):
    image = np.zeros((32, 32, channels), dtype=np.uint8)
    if channels == 4:
      image[:, :, 3] = 255

    landmarks = [
        landmark_module.NormalizedLandmark(x=0.25, y=0.5, z=0.0),
        landmark_module.NormalizedLandmark(x=0.75, y=0.5, z=0.0),
    ]
    drawing_utils.draw_landmarks(
        image,
        landmarks,
        connections=[_Connection(0, 1)],
    )

    # Verify landmark center (8, 16), white landmark border (col < 8), and
    # connection line midpoint (16, 16) were all drawn.
    np.testing.assert_array_equal(image[16, 8, :3], drawing_utils.RED_COLOR)
    self.assertEqual(int(np.max(image[:, :8, 1])), 224)
    np.testing.assert_array_equal(image[16, 16, :3], drawing_utils.WHITE_COLOR)
    if channels == 4:
      np.testing.assert_array_equal(image[:, :, 3], 255)

  @parameterized.parameters(3, 4)
  def test_draw_detection_supports_3_and_4_channel_images(self, channels):
    image = np.zeros((32, 32, channels), dtype=np.uint8)
    if channels == 4:
      image[:, :, 3] = 255

    detection = detections_module.Detection(
        bounding_box=bounding_box_module.BoundingBox(
            origin_x=4, origin_y=4, width=20, height=20
        ),
        categories=[],
        keypoints=[keypoint_module.NormalizedKeypoint(x=0.5, y=0.5)],
    )
    drawing_utils.draw_detection(image, detection)

    # Verify both the keypoint at (16, 16) and the bounding-box border at (4, 4)
    # were drawn onto the image.
    np.testing.assert_array_equal(image[16, 16, :3], drawing_utils.RED_COLOR)
    np.testing.assert_array_equal(image[4, 4, :3], drawing_utils.WHITE_COLOR)
    if channels == 4:
      np.testing.assert_array_equal(image[:, :, 3], 255)

  def test_preserves_explicit_4_tuple_alpha_on_4_channel_image(self):
    image = np.zeros((32, 32, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    landmarks = [landmark_module.NormalizedLandmark(x=0.5, y=0.5, z=0.0)]
    spec = drawing_utils.DrawingSpec(
        color=(10, 20, 30, 128), thickness=-1, circle_radius=2
    )
    drawing_utils.draw_landmarks(
        image,
        landmarks,
        landmark_drawing_spec=spec,
    )
    np.testing.assert_array_equal(image[16, 16], (10, 20, 30, 128))

  @parameterized.parameters(3, 4)
  def test_draw_axis_supports_3_and_4_channel_images(self, channels):
    image = np.zeros((64, 64, channels), dtype=np.uint8)
    if channels == 4:
      image[:, :, 3] = 255

    # Rotate off-axis so X (red), Y (green), and Z (blue) all project to
    # distinct non-zero 2D segments.
    angle = np.float32(np.pi / 4.0)
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    rotation = np.array(
        [[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]],
        dtype=np.float32,
    )
    drawing_utils.draw_axis(
        image,
        rotation=rotation,
        translation=np.array([0.0, 0.0, 0.5], dtype=np.float32),
    )

    # Verify blue z-axis (channel 0), green y-axis (channel 1), and red x-axis
    # (channel 2) were all drawn.
    self.assertGreater(int(np.max(image[:, :, 0])), 0)
    self.assertGreater(int(np.max(image[:, :, 1])), 0)
    self.assertGreater(int(np.max(image[:, :, 2])), 0)
    if channels == 4:
      np.testing.assert_array_equal(image[:, :, 3], 255)

  def test_plot_landmarks_converts_4_tuple_bgra_to_rgba(self):
    landmarks = [
        landmark_module.NormalizedLandmark(x=0.25, y=0.5, z=0.0),
        landmark_module.NormalizedLandmark(x=0.75, y=0.5, z=0.0),
    ]
    mock_ax = mock.MagicMock()
    with (
        mock.patch.object(drawing_utils.plt, 'figure'),
        mock.patch.object(drawing_utils.plt, 'axes', return_value=mock_ax),
        mock.patch.object(drawing_utils.plt, 'show'),
    ):
      drawing_utils.plot_landmarks(
          landmarks,
          connections=[(0, 1)],
          landmark_drawing_spec=drawing_utils.DrawingSpec(
              color=(10, 20, 250, 128), thickness=3
          ),
          connection_drawing_spec=drawing_utils.DrawingSpec(
              color=(30, 40, 200, 64), thickness=2
          ),
      )

    expected_landmark_rgba = (250 / 255.0, 20 / 255.0, 10 / 255.0, 128 / 255.0)
    expected_conn_rgba = (200 / 255.0, 40 / 255.0, 30 / 255.0, 64 / 255.0)
    np.testing.assert_allclose(
        mock_ax.scatter3D.call_args.kwargs['color'], expected_landmark_rgba
    )
    np.testing.assert_allclose(
        mock_ax.plot3D.call_args.kwargs['color'], expected_conn_rgba
    )

  def test_rejects_invalid_channel_counts(self):
    bad_image = np.zeros((16, 16, 2), dtype=np.uint8)
    landmarks = [landmark_module.NormalizedLandmark(x=0.5, y=0.5, z=0.0)]
    with self.assertRaisesRegex(
        ValueError, 'Input image must contain three or four channel'
    ):
      drawing_utils.draw_landmarks(bad_image, landmarks)


if __name__ == '__main__':
  absltest.main()
