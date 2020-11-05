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
"""MediaPipe Hands."""

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
from mediapipe.calculators.tflite import ssd_anchors_calculator_pb2
from mediapipe.calculators.util import association_calculator_pb2
from mediapipe.calculators.util import detections_to_rects_calculator_pb2
from mediapipe.calculators.util import logic_calculator_pb2
from mediapipe.calculators.util import non_max_suppression_calculator_pb2
from mediapipe.calculators.util import rect_transformation_calculator_pb2
from mediapipe.calculators.util import thresholding_calculator_pb2
# pylint: enable=unused-import
from mediapipe.python.solution_base import SolutionBase


class HandLandmark(enum.IntEnum):
  """The 21 hand landmarks."""
  WRIST = 0
  THUMB_CMC = 1
  THUMB_MCP = 2
  THUMB_IP = 3
  THUMB_TIP = 4
  INDEX_FINGER_MCP = 5
  INDEX_FINGER_PIP = 6
  INDEX_FINGER_DIP = 7
  INDEX_FINGER_TIP = 8
  MIDDLE_FINGER_MCP = 9
  MIDDLE_FINGER_PIP = 10
  MIDDLE_FINGER_DIP = 11
  MIDDLE_FINGER_TIP = 12
  RING_FINGER_MCP = 13
  RING_FINGER_PIP = 14
  RING_FINGER_DIP = 15
  RING_FINGER_TIP = 16
  PINKY_MCP = 17
  PINKY_PIP = 18
  PINKY_DIP = 19
  PINKY_TIP = 20


BINARYPB_FILE_PATH = 'mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.binarypb'
HAND_CONNECTIONS = frozenset([
    (HandLandmark.WRIST, HandLandmark.THUMB_CMC),
    (HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP),
    (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP),
    (HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP),
    (HandLandmark.WRIST, HandLandmark.INDEX_FINGER_MCP),
    (HandLandmark.INDEX_FINGER_MCP, HandLandmark.INDEX_FINGER_PIP),
    (HandLandmark.INDEX_FINGER_PIP, HandLandmark.INDEX_FINGER_DIP),
    (HandLandmark.INDEX_FINGER_DIP, HandLandmark.INDEX_FINGER_TIP),
    (HandLandmark.INDEX_FINGER_MCP, HandLandmark.MIDDLE_FINGER_MCP),
    (HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.MIDDLE_FINGER_PIP),
    (HandLandmark.MIDDLE_FINGER_PIP, HandLandmark.MIDDLE_FINGER_DIP),
    (HandLandmark.MIDDLE_FINGER_DIP, HandLandmark.MIDDLE_FINGER_TIP),
    (HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.RING_FINGER_MCP),
    (HandLandmark.RING_FINGER_MCP, HandLandmark.RING_FINGER_PIP),
    (HandLandmark.RING_FINGER_PIP, HandLandmark.RING_FINGER_DIP),
    (HandLandmark.RING_FINGER_DIP, HandLandmark.RING_FINGER_TIP),
    (HandLandmark.RING_FINGER_MCP, HandLandmark.PINKY_MCP),
    (HandLandmark.WRIST, HandLandmark.PINKY_MCP),
    (HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP),
    (HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP),
    (HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP)
])


class Hands(SolutionBase):
  """MediaPipe Hands.

  MediaPipe Hands processes an RGB image and returns the hand landmarks and
  handedness (left v.s. right hand) of each detected hand.

  Note that it determines handedness assuming the input image is mirrored,
  i.e., taken with a front-facing/selfie camera (
  https://en.wikipedia.org/wiki/Front-facing_camera) with images flipped
  horizontally. If that is not the case, use, for instance, cv2.flip(image, 1)
  to flip the image first for a correct handedness output.

  Usage examples:
    import cv2
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # For static images:
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.7)
    for idx, file in enumerate(file_list):
      # Read an image, flip it around y-axis for correct handedness output (see
      # above).
      image = cv2.flip(cv2.imread(file), 1)
      # Convert the BGR image to RGB before processing.
      results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

      # Print handedness and draw hand landmarks on the image.
      print('handedness:', results.multi_handedness)
      if not results.multi_hand_landmarks:
        continue
      annotated_image = image.copy()
      for hand_landmarks in results.multi_hand_landmarks:
        print('hand_landmarks:', hand_landmarks)
        mp_drawing.draw_landmarks(
            annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
      cv2.imwrite(
          '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(image, 1))
    hands.close()

    # For webcam input:
    hands = mp_hands.Hands(
        min_detection_confidence=0.7, min_tracking_confidence=0.5)
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
      results = hands.process(image)

      # Draw the hand annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          mp_drawing.draw_landmarks(
              image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
      cv2.imshow('MediaPipe Hands', image)
      if cv2.waitKey(5) & 0xFF == 27:
        break
    hands.close()
    cap.release()
  """

  def __init__(self,
               static_image_mode=False,
               max_num_hands=2,
               min_detection_confidence=0.7,
               min_tracking_confidence=0.5):
    """Initializes a MediaPipe Hand object.

    Args:
      static_image_mode: If set to False, the solution treats the input images
        as a video stream. It will try to detect hands in the first input
        images, and upon a successful detection further localizes the hand
        landmarks. In subsequent images, once all "max_num_hands" hands are
        detected and the corresponding hand landmarks are localized, it simply
        tracks those landmarks without invoking another detection until it loses
        track of any of the hands. This reduces latency and is ideal for
        processing video frames. If set to True, hand detection runs on every
        input image, ideal for processing a batch of static, possibly unrelated,
        images. Default to False.
      max_num_hands: Maximum number of hands to detect. Default to 2.
      min_detection_confidence: Minimum confidence value ([0.0, 1.0]) from the
        hand detection model for the detection to be considered successful.
        Default to 0.7.
      min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) from the
        landmark-tracking model for the hand landmarks to be considered tracked
        successfully, or otherwise hand detection will be invoked automatically
        on the next input image. Setting it to a higher value can increase
        robustness of the solution, at the expense of a higher latency. Ignored
        if "static_image_mode" is True, where hand detection simply runs on
        every image. Default to 0.5.
    """
    super().__init__(
        binary_graph_path=BINARYPB_FILE_PATH,
        side_inputs={
            'num_hands': max_num_hands,
            'can_skip_detection': not static_image_mode,
        },
        calculator_params={
            'palmdetectioncpu__TensorsToDetectionsCalculator.min_score_thresh':
                min_detection_confidence,
            'handlandmarkcpu__ThresholdingCalculator.threshold':
                min_tracking_confidence,
        },
        outputs=['multi_hand_landmarks', 'multi_handedness'])

  def process(self, image: np.ndarray) -> NamedTuple:
    """Processes an RGB image and returns the hand landmarks and handedness of each detected hand.

    Args:
      image: An RGB image represented as a numpy ndarray.

    Raises:
      RuntimeError: If the underlying graph occurs any error.
      ValueError: If the input image is not three channel RGB.

    Returns:
      A NamedTuple object with two fields: a "multi_hand_landmarks" field that
      contains the hand landmarks on each detected hand and a "multi_handedness"
      field that contains the handedness (left v.s. right hand) of the detected
      hand.
    """

    return super().process(input_data={'image': image})
