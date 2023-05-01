# Copyright 2022 The MediaPipe Authors.
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
"""Gesture recognition constants."""

from mediapipe.model_maker.python.core.utils import file_util

GESTURE_EMBEDDER_KERAS_MODEL_FILES = file_util.DownloadedFiles(
    'gesture_recognizer/gesture_embedder',
    'https://storage.googleapis.com/mediapipe-assets/gesture_embedder.tar.gz',
    is_folder=True,
)
GESTURE_EMBEDDER_TFLITE_MODEL_FILE = file_util.DownloadedFiles(
    'gesture_recognizer/gesture_embedder.tflite',
    'https://storage.googleapis.com/mediapipe-assets/gesture_embedder.tflite',
)
HAND_DETECTOR_TFLITE_MODEL_FILE = file_util.DownloadedFiles(
    'gesture_recognizer/palm_detection_full.tflite',
    'https://storage.googleapis.com/mediapipe-assets/palm_detection_full.tflite',
)
HAND_LANDMARKS_DETECTOR_TFLITE_MODEL_FILE = file_util.DownloadedFiles(
    'gesture_recognizer/hand_landmark_full.tflite',
    'https://storage.googleapis.com/mediapipe-assets/hand_landmark_full.tflite',
)
CANNED_GESTURE_CLASSIFIER_TFLITE_MODEL_FILE = file_util.DownloadedFiles(
    'gesture_recognizer/canned_gesture_classifier.tflite',
    'https://storage.googleapis.com/mediapipe-assets/canned_gesture_classifier.tflite',
)
