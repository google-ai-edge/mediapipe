# Copyright 2022 The MediaPipe Authors.
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

"""MediaPipe Tasks Components Containers API."""

import mediapipe.tasks.python.components.containers.audio_data
import mediapipe.tasks.python.components.containers.bounding_box
import mediapipe.tasks.python.components.containers.category
import mediapipe.tasks.python.components.containers.classification_result
import mediapipe.tasks.python.components.containers.detections
import mediapipe.tasks.python.components.containers.embedding_result
import mediapipe.tasks.python.components.containers.landmark
import mediapipe.tasks.python.components.containers.landmark_detection_result
import mediapipe.tasks.python.components.containers.rect

AudioDataFormat = audio_data.AudioDataFormat
AudioData = audio_data.AudioData
BoundingBox = bounding_box.BoundingBox
Category = category.Category
Classifications = classification_result.Classifications
ClassificationResult = classification_result.ClassificationResult
Detection = detections.Detection
DetectionResult = detections.DetectionResult
Embedding = embedding_result.Embedding
EmbeddingResult = embedding_result.EmbeddingResult
Landmark = landmark.Landmark
NormalizedLandmark = landmark.NormalizedLandmark
LandmarksDetectionResult = landmark_detection_result.LandmarksDetectionResult
Rect = rect.Rect
NormalizedRect = rect.NormalizedRect

# Remove unnecessary modules to avoid duplication in API docs.
del audio_data
del bounding_box
del category
del classification_result
del detections
del embedding_result
del landmark
del landmark_detection_result
del rect
del mediapipe
