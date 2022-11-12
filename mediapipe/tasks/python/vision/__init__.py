# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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

"""MediaPipe Tasks Vision API."""

import mediapipe.tasks.python.vision.core
import mediapipe.tasks.python.vision.gesture_recognizer
import mediapipe.tasks.python.vision.hand_landmarker
import mediapipe.tasks.python.vision.image_classifier
import mediapipe.tasks.python.vision.image_embedder
import mediapipe.tasks.python.vision.image_segmenter
import mediapipe.tasks.python.vision.object_detector

GestureRecognizer = gesture_recognizer.GestureRecognizer
GestureRecognizerOptions = gesture_recognizer.GestureRecognizerOptions
GestureRecognizerResult = gesture_recognizer.GestureRecognizerResult
HandLandmarker = hand_landmarker.HandLandmarker
HandLandmarkerOptions = hand_landmarker.HandLandmarkerOptions
HandLandmarkerResult = hand_landmarker.HandLandmarkerResult
ImageClassifier = image_classifier.ImageClassifier
ImageClassifierOptions = image_classifier.ImageClassifierOptions
ImageClassifierResult = image_classifier.ImageClassifierResult
ImageEmbedder = image_embedder.ImageEmbedder
ImageEmbedderOptions = image_embedder.ImageEmbedderOptions
ImageEmbedderResult = image_embedder.ImageEmbedderResult
ImageSegmenter = image_segmenter.ImageSegmenter
ImageSegmenterOptions = image_segmenter.ImageSegmenterOptions
ObjectDetector = object_detector.ObjectDetector
ObjectDetectorOptions = object_detector.ObjectDetectorOptions
RunningMode = core.vision_task_running_mode.VisionTaskRunningMode

# Remove unnecessary modules to avoid duplication in API docs.
del core
del gesture_recognizer
del hand_landmarker
del image_classifier
del image_embedder
del image_segmenter
del object_detector
del mediapipe
