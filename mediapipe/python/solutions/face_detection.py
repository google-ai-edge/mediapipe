# Copyright 2021 The MediaPipe Authors.
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
"""MediaPipe Face Detection."""

import enum
from typing import NamedTuple, Union

import numpy as np
from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import location_data_pb2
# pylint: disable=unused-import
from mediapipe.calculators.tensor import image_to_tensor_calculator_pb2
from mediapipe.calculators.tensor import inference_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_detections_calculator_pb2
from mediapipe.calculators.tflite import ssd_anchors_calculator_pb2
from mediapipe.calculators.util import non_max_suppression_calculator_pb2
# pylint: enable=unused-import
from mediapipe.python.solution_base import SolutionBase

_SHORT_RANGE_GRAPH_FILE_PATH = 'mediapipe/modules/face_detection/face_detection_short_range_cpu.binarypb'
_FULL_RANGE_GRAPH_FILE_PATH = 'mediapipe/modules/face_detection/face_detection_full_range_cpu.binarypb'


def get_key_point(
    detection: detection_pb2.Detection, key_point_enum: 'FaceKeyPoint'
) -> Union[None, location_data_pb2.LocationData.RelativeKeypoint]:
  """A convenience method to return a face key point by the FaceKeyPoint type.

  Args:
    detection: A detection proto message that contains face key points.
    key_point_enum: A FaceKeyPoint type.

  Returns:
    A RelativeKeypoint proto message.
  """
  if not detection or not detection.location_data:
    return None
  return detection.location_data.relative_keypoints[key_point_enum]


class FaceKeyPoint(enum.IntEnum):
  """The enum type of the six face detection key points."""
  RIGHT_EYE = 0
  LEFT_EYE = 1
  NOSE_TIP = 2
  MOUTH_CENTER = 3
  RIGHT_EAR_TRAGION = 4
  LEFT_EAR_TRAGION = 5


class FaceDetection(SolutionBase):
  """MediaPipe Face Detection.

  MediaPipe Face Detection processes an RGB image and returns a list of the
  detected face location data.

  Please refer to
  https://solutions.mediapipe.dev/face_detection#python-solution-api
  for usage examples.
  """

  def __init__(self, min_detection_confidence=0.5, model_selection=0):
    """Initializes a MediaPipe Face Detection object.

    Args:
      min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for face
        detection to be considered successful. See details in
        https://solutions.mediapipe.dev/face_detection#min_detection_confidence.
      model_selection: 0 or 1. 0 to select a short-range model that works
        best for faces within 2 meters from the camera, and 1 for a full-range
        model best for faces within 5 meters. See details in
        https://solutions.mediapipe.dev/face_detection#model_selection.
    """

    binary_graph_path = _FULL_RANGE_GRAPH_FILE_PATH if model_selection == 1 else _SHORT_RANGE_GRAPH_FILE_PATH
    subgraph_name = 'facedetectionfullrangecommon' if model_selection == 1 else 'facedetectionshortrangecommon'

    super().__init__(
        binary_graph_path=binary_graph_path,
        calculator_params={
            subgraph_name + '__TensorsToDetectionsCalculator.min_score_thresh':
                min_detection_confidence,
        },
        outputs=['detections'])

  def process(self, image: np.ndarray) -> NamedTuple:
    """Processes an RGB image and returns a list of the detected face location data.

    Args:
      image: An RGB image represented as a numpy ndarray.

    Raises:
      RuntimeError: If the underlying graph throws any error.
      ValueError: If the input image is not three channel RGB.

    Returns:
      A NamedTuple object with a "detections" field that contains a list of the
      detected face location data.
    """

    return super().process(input_data={'image': image})
