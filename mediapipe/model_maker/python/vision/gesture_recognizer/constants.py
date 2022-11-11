# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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

GESTURE_EMBEDDER_KERAS_MODEL_PATH = 'mediapipe/model_maker/models/gesture_recognizer/gesture_embedder'
GESTURE_EMBEDDER_TFLITE_MODEL_FILE = 'mediapipe/model_maker/models/gesture_recognizer/gesture_embedder.tflite'
HAND_DETECTOR_TFLITE_MODEL_FILE = 'mediapipe/model_maker/models/gesture_recognizer/palm_detection_full.tflite'
HAND_LANDMARKS_DETECTOR_TFLITE_MODEL_FILE = 'mediapipe/model_maker/models/gesture_recognizer/hand_landmark_full.tflite'
CANNED_GESTURE_CLASSIFIER_TFLITE_MODEL_FILE = 'mediapipe/model_maker/models/gesture_recognizer/canned_gesture_classifier.tflite'
