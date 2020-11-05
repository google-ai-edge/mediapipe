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
"""MediaPipe FaceMesh."""

from typing import NamedTuple

import numpy as np

# pylint: disable=unused-import
from mediapipe.calculators.core import gate_calculator_pb2
from mediapipe.calculators.core import split_vector_calculator_pb2
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

BINARYPB_FILE_PATH = 'mediapipe/modules/face_landmark/face_landmark_front_cpu.binarypb'
FACE_CONNECTIONS = frozenset([
    # Lips.
    (61, 146),
    (146, 91),
    (91, 181),
    (181, 84),
    (84, 17),
    (17, 314),
    (314, 405),
    (405, 321),
    (321, 375),
    (375, 291),
    (61, 185),
    (185, 40),
    (40, 39),
    (39, 37),
    (37, 0),
    (0, 267),
    (267, 269),
    (269, 270),
    (270, 409),
    (409, 291),
    (78, 95),
    (95, 88),
    (88, 178),
    (178, 87),
    (87, 14),
    (14, 317),
    (317, 402),
    (402, 318),
    (318, 324),
    (324, 308),
    (78, 191),
    (191, 80),
    (80, 81),
    (81, 82),
    (82, 13),
    (13, 312),
    (312, 311),
    (311, 310),
    (310, 415),
    (415, 308),
    # Left eye.
    (33, 7),
    (7, 163),
    (163, 144),
    (144, 145),
    (145, 153),
    (153, 154),
    (154, 155),
    (155, 133),
    (33, 246),
    (246, 161),
    (161, 160),
    (160, 159),
    (159, 158),
    (158, 157),
    (157, 173),
    (173, 133),
    # Left eyebrow.
    (46, 53),
    (53, 52),
    (52, 65),
    (65, 55),
    (70, 63),
    (63, 105),
    (105, 66),
    (66, 107),
    # Right eye.
    (263, 249),
    (249, 390),
    (390, 373),
    (373, 374),
    (374, 380),
    (380, 381),
    (381, 382),
    (382, 362),
    (263, 466),
    (466, 388),
    (388, 387),
    (387, 386),
    (386, 385),
    (385, 384),
    (384, 398),
    (398, 362),
    # Right eyebrow.
    (276, 283),
    (283, 282),
    (282, 295),
    (295, 285),
    (300, 293),
    (293, 334),
    (334, 296),
    (296, 336),
    # Face oval.
    (10, 338),
    (338, 297),
    (297, 332),
    (332, 284),
    (284, 251),
    (251, 389),
    (389, 356),
    (356, 454),
    (454, 323),
    (323, 361),
    (361, 288),
    (288, 397),
    (397, 365),
    (365, 379),
    (379, 378),
    (378, 400),
    (400, 377),
    (377, 152),
    (152, 148),
    (148, 176),
    (176, 149),
    (149, 150),
    (150, 136),
    (136, 172),
    (172, 58),
    (58, 132),
    (132, 93),
    (93, 234),
    (234, 127),
    (127, 162),
    (162, 21),
    (21, 54),
    (54, 103),
    (103, 67),
    (67, 109),
    (109, 10)
])


class FaceMesh(SolutionBase):
  """MediaPipe FaceMesh.

  MediaPipe FaceMesh processes an RGB image and returns the face landmarks on
  each detected face.

  Usage examples:
    import cv2
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    # For static images:
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5)
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    for idx, file in enumerate(file_list):
      image = cv2.imread(file)
      # Convert the BGR image to RGB before processing.
      results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

      # Print and draw face mesh landmarks on the image.
      if not results.multi_face_landmarks:
        continue
      annotated_image = image.copy()
      for face_landmarks in results.multi_face_landmarks:
        print('face_landmarks:', face_landmarks)
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
      cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', image)
    face_mesh.close()

    # For webcam input:
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
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
      results = face_mesh.process(image)

      # Draw the face mesh annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
          mp_drawing.draw_landmarks(
              image=image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACE_CONNECTIONS,
              landmark_drawing_spec=drawing_spec,
              connection_drawing_spec=drawing_spec)
      cv2.imshow('MediaPipe FaceMesh', image)
      if cv2.waitKey(5) & 0xFF == 27:
        break
    face_mesh.close()
    cap.release()
  """

  def __init__(self,
               static_image_mode=False,
               max_num_faces=2,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
    """Initializes a MediaPipe FaceMesh object.

    Args:
      static_image_mode: If set to False, the solution treats the input images
        as a video stream. It will try to detect faces in the first input
        images, and upon a successful detection further localizes the face
        landmarks. In subsequent images, once all "max_num_faces" faces are
        detected and the corresponding face landmarks are localized, it simply
        tracks those landmarks without invoking another detection until it loses
        track of any of the faces. This reduces latency and is ideal for
        processing video frames. If set to True, face detection runs on every
        input image, ideal for processing a batch of static, possibly unrelated,
        images. Default to False.
      max_num_faces: Maximum number of faces to detect. Default to 2.
      min_detection_confidence: Minimum confidence value ([0.0, 1.0]) from the
        face detection model for the detection to be considered successful.
        Default to 0.5.
      min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) from the
        landmark-tracking model for the face landmarks to be considered tracked
        successfully, or otherwise face detection will be invoked automatically
        on the next input image. Setting it to a higher value can increase
        robustness of the solution, at the expense of a higher latency. Ignored
        if "static_image_mode" is True, where face detection simply runs on
        every image. Default to 0.5.
    """
    super().__init__(
        binary_graph_path=BINARYPB_FILE_PATH,
        side_inputs={
            'num_faces': max_num_faces,
            'can_skip_detection': not static_image_mode,
        },
        calculator_params={
            'facedetectionfrontcpu__TensorsToDetectionsCalculator.min_score_thresh':
                min_detection_confidence,
            'facelandmarkcpu__ThresholdingCalculator.threshold':
                min_tracking_confidence,
        },
        outputs=['multi_face_landmarks'])

  def process(self, image: np.ndarray) -> NamedTuple:
    """Processes an RGB image and returns the face landmarks on each detected face.

    Args:
      image: An RGB image represented as a numpy ndarray.

    Raises:
      RuntimeError: If the underlying graph occurs any error.
      ValueError: If the input image is not three channel RGB.

    Returns:
      A NamedTuple object with a "multi_face_landmarks" field that contains the
      face landmarks on each detected face.
    """

    return super().process(input_data={'image': image})
