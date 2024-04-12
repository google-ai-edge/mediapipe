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

"""MediaPipe Tasks Vision API."""

import mediapipe.tasks.python.vision.core
import mediapipe.tasks.python.vision.face_aligner
import mediapipe.tasks.python.vision.face_detector
import mediapipe.tasks.python.vision.face_landmarker
import mediapipe.tasks.python.vision.face_stylizer
import mediapipe.tasks.python.vision.gesture_recognizer
import mediapipe.tasks.python.vision.hand_landmarker
import mediapipe.tasks.python.vision.holistic_landmarker
import mediapipe.tasks.python.vision.image_classifier
import mediapipe.tasks.python.vision.image_embedder
import mediapipe.tasks.python.vision.image_segmenter
import mediapipe.tasks.python.vision.interactive_segmenter
import mediapipe.tasks.python.vision.object_detector
import mediapipe.tasks.python.vision.pose_landmarker

FaceAligner = face_aligner.FaceAligner
FaceAlignerOptions = face_aligner.FaceAlignerOptions
FaceDetector = face_detector.FaceDetector
FaceDetectorOptions = face_detector.FaceDetectorOptions
FaceDetectorResult = face_detector.FaceDetectorResult
FaceLandmarker = face_landmarker.FaceLandmarker
FaceLandmarkerOptions = face_landmarker.FaceLandmarkerOptions
FaceLandmarkerResult = face_landmarker.FaceLandmarkerResult
FaceLandmarksConnections = face_landmarker.FaceLandmarksConnections
FaceStylizer = face_stylizer.FaceStylizer
FaceStylizerOptions = face_stylizer.FaceStylizerOptions
GestureRecognizer = gesture_recognizer.GestureRecognizer
GestureRecognizerOptions = gesture_recognizer.GestureRecognizerOptions
GestureRecognizerResult = gesture_recognizer.GestureRecognizerResult
HandLandmarker = hand_landmarker.HandLandmarker
HandLandmarkerOptions = hand_landmarker.HandLandmarkerOptions
HandLandmarkerResult = hand_landmarker.HandLandmarkerResult
HandLandmarksConnections = hand_landmarker.HandLandmarksConnections
ImageClassifier = image_classifier.ImageClassifier
ImageClassifierOptions = image_classifier.ImageClassifierOptions
ImageClassifierResult = image_classifier.ImageClassifierResult
ImageEmbedder = image_embedder.ImageEmbedder
ImageEmbedderOptions = image_embedder.ImageEmbedderOptions
ImageEmbedderResult = image_embedder.ImageEmbedderResult
ImageSegmenter = image_segmenter.ImageSegmenter
ImageSegmenterOptions = image_segmenter.ImageSegmenterOptions
ImageProcessingOptions = core.image_processing_options.ImageProcessingOptions
InteractiveSegmenter = interactive_segmenter.InteractiveSegmenter
InteractiveSegmenterOptions = interactive_segmenter.InteractiveSegmenterOptions
InteractiveSegmenterRegionOfInterest = interactive_segmenter.RegionOfInterest
ObjectDetector = object_detector.ObjectDetector
ObjectDetectorOptions = object_detector.ObjectDetectorOptions
ObjectDetectorResult = object_detector.ObjectDetectorResult
PoseLandmarker = pose_landmarker.PoseLandmarker
PoseLandmarkerOptions = pose_landmarker.PoseLandmarkerOptions
PoseLandmarkerResult = pose_landmarker.PoseLandmarkerResult
PoseLandmarksConnections = pose_landmarker.PoseLandmarksConnections
HolisticLandmarker = holistic_landmarker.HolisticLandmarker
HolisticLandmarkerOptions = holistic_landmarker.HolisticLandmarkerOptions
HolisticLandmarkerResult = holistic_landmarker.HolisticLandmarkerResult

RunningMode = core.vision_task_running_mode.VisionTaskRunningMode

# Remove unnecessary modules to avoid duplication in API docs.
del core
del face_aligner
del face_detector
del face_landmarker
del face_stylizer
del gesture_recognizer
del hand_landmarker
del holistic_landmarker
del image_classifier
del image_embedder
del image_segmenter
del interactive_segmenter
del object_detector
del pose_landmarker
del mediapipe
