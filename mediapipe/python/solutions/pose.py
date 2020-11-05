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

# Lint as: python3
"""MediaPipe Pose."""

import enum
from typing import NamedTuple

import numpy as np

# pylint: disable=unused-import
from mediapipe.calculators.core import gate_calculator_pb2
from mediapipe.calculators.core import split_vector_calculator_pb2
from mediapipe.calculators.tensor import image_to_tensor_calculator_pb2
from mediapipe.calculators.tensor import inference_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_classification_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_detections_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_landmarks_calculator_pb2
from mediapipe.calculators.util import detections_to_rects_calculator_pb2
from mediapipe.calculators.util import landmarks_smoothing_calculator_pb2
from mediapipe.calculators.util import logic_calculator_pb2
from mediapipe.calculators.util import non_max_suppression_calculator_pb2
from mediapipe.calculators.util import rect_transformation_calculator_pb2
from mediapipe.calculators.util import thresholding_calculator_pb2
# pylint: enable=unused-import
from mediapipe.python.solution_base import SolutionBase


class PoseLandmark(enum.IntEnum):
  """The 25 (upper-body) pose landmarks."""
  NOSE = 0
  RIGHT_EYE_INNER = 1
  RIGHT_EYE = 2
  RIGHT_EYE_OUTER = 3
  LEFT_EYE_INNER = 4
  LEFT_EYE = 5
  LEFT_EYE_OUTER = 6
  RIGHT_EAR = 7
  LEFT_EAR = 8
  MOUTH_RIGHT = 9
  MOUTH_LEFT = 10
  RIGHT_SHOULDER = 11
  LEFT_SHOULDER = 12
  RIGHT_ELBOW = 13
  LEFT_ELBOW = 14
  RIGHT_WRIST = 15
  LEFT_WRIST = 16
  RIGHT_PINKY = 17
  LEFT_PINKY = 18
  RIGHT_INDEX = 19
  LEFT_INDEX = 20
  RIGHT_THUMB = 21
  LEFT_THUMB = 22
  RIGHT_HIP = 23
  LEFT_HIP = 24


BINARYPB_FILE_PATH = 'mediapipe/modules/pose_landmark/pose_landmark_upper_body_smoothed_cpu.binarypb'
POSE_CONNECTIONS = frozenset([
    (PoseLandmark.NOSE, PoseLandmark.RIGHT_EYE_INNER),
    (PoseLandmark.RIGHT_EYE_INNER, PoseLandmark.RIGHT_EYE),
    (PoseLandmark.RIGHT_EYE, PoseLandmark.RIGHT_EYE_OUTER),
    (PoseLandmark.RIGHT_EYE_OUTER, PoseLandmark.RIGHT_EAR),
    (PoseLandmark.NOSE, PoseLandmark.LEFT_EYE_INNER),
    (PoseLandmark.LEFT_EYE_INNER, PoseLandmark.LEFT_EYE),
    (PoseLandmark.LEFT_EYE, PoseLandmark.LEFT_EYE_OUTER),
    (PoseLandmark.LEFT_EYE_OUTER, PoseLandmark.LEFT_EAR),
    (PoseLandmark.MOUTH_RIGHT, PoseLandmark.MOUTH_LEFT),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.LEFT_SHOULDER),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
    (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_PINKY),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_INDEX),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_THUMB),
    (PoseLandmark.RIGHT_PINKY, PoseLandmark.RIGHT_INDEX),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
    (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_PINKY),
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_INDEX),
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_THUMB),
    (PoseLandmark.LEFT_PINKY, PoseLandmark.LEFT_INDEX),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
    (PoseLandmark.RIGHT_HIP, PoseLandmark.LEFT_HIP)
])


class Pose(SolutionBase):
  """MediaPipe Pose.

  MediaPipe Pose processes an RGB image and returns pose landmarks on the most
  prominent person detected.

  Usage examples:
    import cv2
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # For static images:
    pose = mp_pose.Pose(
        static_image_mode=True, min_detection_confidence=0.5)
    for idx, file in enumerate(file_list):
      image = cv2.imread(file)
      # Convert the BGR image to RGB before processing.
      results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

      # Print and draw pose landmarks on the image.
      print(
          'nose landmark:',
           results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE])
      annotated_image = image.copy()
      mp_drawing.draw_landmarks(
          annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
      cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', image)
    pose.close()

    # For webcam input:
    pose = mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        break

      # Flip the image horizontally for a later selfie-view display, and convert
      # the BGR image to RGB.
      image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      results = pose.process(image)

      # Draw the pose annotation on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      mp_drawing.draw_landmarks(
          image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
      cv2.imshow('MediaPipe Pose', image)
      if cv2.waitKey(5) & 0xFF == 27:
        break
    pose.close()
    cap.release()
  """

  def __init__(self,
               static_image_mode=False,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
    """Initializes a MediaPipe Pose object.

    Args:
      static_image_mode: If set to False, the solution treats the input images
        as a video stream. It will try to detect the most prominent person in
        the very first images, and upon a successful detection further localizes
        the pose landmarks. In subsequent images, it then simply tracks those
        landmarks without invoking another detection until it loses track, on
        reducing computation and latency. If set to True, person detection runs
        every input image, ideal for processing a batch of static, possibly
        unrelated, images. Default to False.
      min_detection_confidence: Minimum confidence value ([0.0, 1.0]) from the
        person-detection model for the detection to be considered successful.
        Default to 0.5.
      min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) from the
        landmark-tracking model for the pose landmarks to be considered tracked
        successfully, or otherwise person detection will be invoked
        automatically on the next input image. Setting it to a higher value can
        increase robustness of the solution, at the expense of a higher latency.
        Ignored if "static_image_mode" is True, where person detection simply
        runs on every image. Default to 0.5.
    """
    super().__init__(
        binary_graph_path=BINARYPB_FILE_PATH,
        side_inputs={
            'can_skip_detection': not static_image_mode,
        },
        calculator_params={
            'poselandmarkupperbodycpu__posedetectioncpu__TensorsToDetectionsCalculator.min_score_thresh':
                min_detection_confidence,
            'poselandmarkupperbodycpu__poselandmarkupperbodybyroicpu__ThresholdingCalculator.threshold':
                min_tracking_confidence,
        },
        outputs=['pose_landmarks'])

  def process(self, image: np.ndarray) -> NamedTuple:
    """Processes an RGB image and returns the pose landmarks on the most prominent person detected.

    Args:
      image: An RGB image represented as a numpy ndarray.

    Raises:
      RuntimeError: If the underlying graph occurs any error.
      ValueError: If the input image is not three channel RGB.

    Returns:
      A NamedTuple object with a "pose_landmarks" field that contains the pose
      landmarks on the most prominent person detected.
    """

    return super().process(input_data={'image': image})
