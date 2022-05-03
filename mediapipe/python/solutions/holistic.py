# Copyright 2020-2021 The MediaPipe Authors.
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
"""MediaPipe Holistic."""

from typing import NamedTuple

import numpy as np

# The following imports are needed because python pb2 silently discards
# unknown protobuf fields.
# pylint: disable=unused-import
from mediapipe.calculators.core import constant_side_packet_calculator_pb2
from mediapipe.calculators.core import gate_calculator_pb2
from mediapipe.calculators.core import split_vector_calculator_pb2
from mediapipe.calculators.tensor import image_to_tensor_calculator_pb2
from mediapipe.calculators.tensor import inference_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_classification_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_floats_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_landmarks_calculator_pb2
from mediapipe.calculators.tflite import ssd_anchors_calculator_pb2
from mediapipe.calculators.util import detections_to_rects_calculator_pb2
from mediapipe.calculators.util import landmark_projection_calculator_pb2
from mediapipe.calculators.util import local_file_contents_calculator_pb2
from mediapipe.calculators.util import non_max_suppression_calculator_pb2
from mediapipe.calculators.util import rect_transformation_calculator_pb2
from mediapipe.framework.tool import switch_container_pb2
from mediapipe.modules.holistic_landmark.calculators import roi_tracking_calculator_pb2
# pylint: enable=unused-import

from mediapipe.python.solution_base import SolutionBase
from mediapipe.python.solutions import download_utils
# pylint: disable=unused-import
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION
from mediapipe.python.solutions.hands import HandLandmark
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
# pylint: enable=unused-import

_BINARYPB_FILE_PATH = 'mediapipe/modules/holistic_landmark/holistic_landmark_cpu.binarypb'


def _download_oss_pose_landmark_model(model_complexity):
  """Downloads the pose landmark lite/heavy model from the MediaPipe Github repo if it doesn't exist in the package."""

  if model_complexity == 0:
    download_utils.download_oss_model(
        'mediapipe/modules/pose_landmark/pose_landmark_lite.tflite')
  elif model_complexity == 2:
    download_utils.download_oss_model(
        'mediapipe/modules/pose_landmark/pose_landmark_heavy.tflite')


class Holistic(SolutionBase):
  """MediaPipe Holistic.

  MediaPipe Holistic processes an RGB image and returns pose landmarks, left and
  right hand landmarks, and face mesh landmarks on the most prominent person
  detected.

  Please refer to https://solutions.mediapipe.dev/holistic#python-solution-api
  for usage examples.
  """

  def __init__(self,
               static_image_mode=False,
               model_complexity=1,
               smooth_landmarks=True,
               enable_segmentation=False,
               smooth_segmentation=True,
               refine_face_landmarks=False,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
    """Initializes a MediaPipe Holistic object.

    Args:
      static_image_mode: Whether to treat the input images as a batch of static
        and possibly unrelated images, or a video stream. See details in
        https://solutions.mediapipe.dev/holistic#static_image_mode.
      model_complexity: Complexity of the pose landmark model: 0, 1 or 2. See
        details in https://solutions.mediapipe.dev/holistic#model_complexity.
      smooth_landmarks: Whether to filter landmarks across different input
        images to reduce jitter. See details in
        https://solutions.mediapipe.dev/holistic#smooth_landmarks.
      enable_segmentation: Whether to predict segmentation mask. See details in
        https://solutions.mediapipe.dev/holistic#enable_segmentation.
      smooth_segmentation: Whether to filter segmentation across different input
        images to reduce jitter. See details in
        https://solutions.mediapipe.dev/holistic#smooth_segmentation.
      refine_face_landmarks: Whether to further refine the landmark coordinates
        around the eyes and lips, and output additional landmarks around the
        irises. Default to False. See details in
        https://solutions.mediapipe.dev/holistic#refine_face_landmarks.
      min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for person
        detection to be considered successful. See details in
        https://solutions.mediapipe.dev/holistic#min_detection_confidence.
      min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the
        pose landmarks to be considered tracked successfully. See details in
        https://solutions.mediapipe.dev/holistic#min_tracking_confidence.
    """
    _download_oss_pose_landmark_model(model_complexity)
    super().__init__(
        binary_graph_path=_BINARYPB_FILE_PATH,
        side_inputs={
            'model_complexity': model_complexity,
            'smooth_landmarks': smooth_landmarks and not static_image_mode,
            'enable_segmentation': enable_segmentation,
            'smooth_segmentation':
                smooth_segmentation and not static_image_mode,
            'refine_face_landmarks': refine_face_landmarks,
            'use_prev_landmarks': not static_image_mode,
        },
        calculator_params={
            'poselandmarkcpu__posedetectioncpu__TensorsToDetectionsCalculator.min_score_thresh':
                min_detection_confidence,
            'poselandmarkcpu__poselandmarkbyroicpu__tensorstoposelandmarksandsegmentation__ThresholdingCalculator.threshold':
                min_tracking_confidence,
        },
        outputs=[
            'pose_landmarks', 'pose_world_landmarks', 'left_hand_landmarks',
            'right_hand_landmarks', 'face_landmarks', 'segmentation_mask'
        ])

  def process(self, image: np.ndarray) -> NamedTuple:
    """Processes an RGB image and returns the pose landmarks, left and right hand landmarks, and face landmarks on the most prominent person detected.

    Args:
      image: An RGB image represented as a numpy ndarray.

    Raises:
      RuntimeError: If the underlying graph throws any error.
      ValueError: If the input image is not three channel RGB.

    Returns:
      A NamedTuple with fields describing the landmarks on the most prominate
      person detected:
        1) "pose_landmarks" field that contains the pose landmarks.
        2) "pose_world_landmarks" field that contains the pose landmarks in
        real-world 3D coordinates that are in meters with the origin at the
        center between hips.
        3) "left_hand_landmarks" field that contains the left-hand landmarks.
        4) "right_hand_landmarks" field that contains the right-hand landmarks.
        5) "face_landmarks" field that contains the face landmarks.
        6) "segmentation_mask" field that contains the segmentation mask if
           "enable_segmentation" is set to true.
    """

    results = super().process(input_data={'image': image})
    if results.pose_landmarks:  # pytype: disable=attribute-error
      for landmark in results.pose_landmarks.landmark:  # pytype: disable=attribute-error
        landmark.ClearField('presence')
    if results.pose_world_landmarks:  # pytype: disable=attribute-error
      for landmark in results.pose_world_landmarks.landmark:  # pytype: disable=attribute-error
        landmark.ClearField('presence')
    return results
