# Copyright 2025 The MediaPipe Authors.
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
"""Proto loading utilities for vision tests."""

from typing import List

from mediapipe.framework.formats import classification_pb2
from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.framework.formats import location_data_pb2
from mediapipe.tasks.cc.components.containers.proto import landmarks_detection_result_pb2
from mediapipe.tasks.python.components.containers import bounding_box as bounding_box_module
from mediapipe.tasks.python.components.containers import category as category_module
from mediapipe.tasks.python.components.containers import detections as detections_module
from mediapipe.tasks.python.components.containers import keypoint as keypoint_module
from mediapipe.tasks.python.components.containers import landmark as landmark_module
from mediapipe.tasks.python.components.containers import landmark_detection_result as landmark_detection_result_module
from mediapipe.tasks.python.components.containers import rect as rect_module
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision import pose_landmarker

PoseLandmarkerResult = pose_landmarker.PoseLandmarkerResult
HandLandmarkerResult = hand_landmarker.HandLandmarkerResult
_LandmarksDetectionResultProto = (
    landmarks_detection_result_pb2.LandmarksDetectionResult
)
_NormalizedRect = rect_module.NormalizedRect


def create_bounding_box_from_proto(
    pb2_obj: location_data_pb2.LocationData.BoundingBox,
) -> bounding_box_module.BoundingBox:
  return bounding_box_module.BoundingBox(
      origin_x=pb2_obj.xmin,
      origin_y=pb2_obj.ymin,
      width=pb2_obj.width,
      height=pb2_obj.height,
  )


def create_category_from_proto(
    pb2_obj: classification_pb2.Classification,
) -> category_module.Category:
  return category_module.Category(
      index=pb2_obj.index,
      score=pb2_obj.score,
      display_name=pb2_obj.display_name,
      category_name=pb2_obj.label,
  )


def create_normalized_keypoint_from_proto(
    pb2_obj: location_data_pb2.LocationData.RelativeKeypoint,
) -> keypoint_module.NormalizedKeypoint:
  return keypoint_module.NormalizedKeypoint(
      x=pb2_obj.x,
      y=pb2_obj.y,
      label=pb2_obj.keypoint_label,
      score=pb2_obj.score,
  )


def create_detection_from_proto(
    pb2_obj: detection_pb2.Detection,
) -> detections_module.Detection:
  """Creates a `Detection` from the given protobuf object."""
  categories = [
      create_category_from_proto(
          classification_pb2.Classification(
              score=score,
              index=(
                  pb2_obj.label_id[idx] if idx < len(pb2_obj.label_id) else None
              ),
              label=pb2_obj.label[idx] if idx < len(pb2_obj.label) else None,
              display_name=(
                  pb2_obj.display_name[idx]
                  if idx < len(pb2_obj.display_name)
                  else None
              ),
          )
      )
      for idx, score in enumerate(pb2_obj.score)
  ]

  if pb2_obj.location_data.relative_keypoints:
    keypoints = [
        create_normalized_keypoint_from_proto(elem)
        for elem in pb2_obj.location_data.relative_keypoints
    ]
  else:
    keypoints = []

  return detections_module.Detection(
      bounding_box=create_bounding_box_from_proto(
          pb2_obj.location_data.bounding_box
      ),
      categories=categories,
      keypoints=keypoints,
  )


def create_landmark_from_proto(
    pb2_obj: landmark_pb2.Landmark,
) -> landmark_module.Landmark:
  return landmark_module.Landmark(
      x=pb2_obj.x,
      y=pb2_obj.y,
      z=pb2_obj.z,
      visibility=pb2_obj.visibility if pb2_obj.HasField('visibility') else None,
      presence=pb2_obj.presence if pb2_obj.HasField('presence') else None,
  )


def create_normalized_landmark_from_proto(
    pb2_obj: landmark_pb2.NormalizedLandmark,
) -> landmark_module.NormalizedLandmark:
  return landmark_module.NormalizedLandmark(
      x=pb2_obj.x,
      y=pb2_obj.y,
      z=pb2_obj.z,
      visibility=pb2_obj.visibility if pb2_obj.HasField('visibility') else None,
      presence=pb2_obj.presence if pb2_obj.HasField('presence') else None,
  )


def create_landmarks_detection_result_from_proto(
    pb2_obj: landmarks_detection_result_pb2.LandmarksDetectionResult,
) -> landmark_detection_result_module.LandmarksDetectionResult:
  """Creates a `LandmarksDetectionResult` from the given protobuf object."""
  if pb2_obj.HasField('landmarks'):
    landmarks = [
        create_normalized_landmark_from_proto(landmark_proto)
        for landmark_proto in pb2_obj.landmarks.landmark
    ]
  else:
    landmarks = []

  if pb2_obj.HasField('classifications'):
    categories = [
        create_category_from_proto(category_proto)
        for category_proto in pb2_obj.classifications.classification
    ]
  else:
    categories = []

  return landmark_detection_result_module.LandmarksDetectionResult(
      landmarks=landmarks,
      world_landmarks=[],
      categories=categories,
      rect=_NormalizedRect(x_center=0.0, y_center=0.0, width=0.0, height=0.0),
  )


def create_normalized_landmark_list_from_proto(
    pb2_obj: landmark_pb2.NormalizedLandmarkList,
) -> List[landmark_module.NormalizedLandmark]:
  """Creates a list of `NormalizedLandmark` from the given protobuf object."""
  return [
      create_normalized_landmark_from_proto(landmark)
      for landmark in pb2_obj.landmark
  ]


def create_classification_list_from_proto(
    pb2_obj: classification_pb2.ClassificationList,
) -> List[category_module.Category]:
  """Creates a list of `Category` from the given protobuf object."""
  return [
      create_category_from_proto(classification)
      for classification in pb2_obj.classification
  ]
