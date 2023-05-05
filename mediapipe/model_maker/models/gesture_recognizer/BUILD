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

load(
    "//mediapipe/framework/tool:mediapipe_files.bzl",
    "mediapipe_files",
)

licenses(["notice"])

package(default_visibility = ["//mediapipe/model_maker/python/vision/gesture_recognizer:__subpackages__"])

mediapipe_files(
    srcs = [
        "canned_gesture_classifier.tflite",
        "gesture_embedder.tflite",
        "gesture_embedder/keras_metadata.pb",
        "gesture_embedder/saved_model.pb",
        "gesture_embedder/variables/variables.data-00000-of-00001",
        "gesture_embedder/variables/variables.index",
        "hand_landmark_full.tflite",
        "palm_detection_full.tflite",
    ],
)
