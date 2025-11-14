# Copyright 2025 The MediaPipe Authors.
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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

exports_files(["mediapipe_tasks_c_version_script.lds"])

# bazel build -c opt --linkopt -s --strip always --define MEDIAPIPE_DISABLE_GPU=1 \
# //mediapipe/tasks/c:libmediapipe.so
cc_binary(
    name = "libmediapipe.so",
    linkopts = [
        "-Wl,-soname=libmediapipe.so",
        "-Wl,--version-script=$(location :mediapipe_tasks_c_version_script.lds)",
        "-fvisibility=hidden",
    ],
    linkshared = True,
    linkstatic = True,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        ":mediapipe_tasks_c_version_script.lds",
        "//mediapipe/tasks/c/audio/audio_classifier:audio_classifier_c_lib",
        "//mediapipe/tasks/c/text/language_detector:language_detector_c_lib",
        "//mediapipe/tasks/c/text/text_classifier:text_classifier_c_lib",
        "//mediapipe/tasks/c/text/text_embedder:text_embedder_c_lib",
        "//mediapipe/tasks/c/vision/core:image_c_lib",
        "//mediapipe/tasks/c/vision/face_detector:face_detector_c_lib",
        "//mediapipe/tasks/c/vision/face_landmarker:face_landmarker_c_lib",
        "//mediapipe/tasks/c/vision/gesture_recognizer:gesture_recognizer_c_lib",
        "//mediapipe/tasks/c/vision/hand_landmarker:hand_landmarker_c_lib",
        "//mediapipe/tasks/c/vision/image_classifier:image_classifier_c_lib",
        "//mediapipe/tasks/c/vision/image_embedder:image_embedder_c_lib",
        "//mediapipe/tasks/c/vision/image_segmenter:image_segmenter_c_lib",
        "//mediapipe/tasks/c/vision/interactive_segmenter:interactive_segmenter_c_lib",
        "//mediapipe/tasks/c/vision/object_detector:object_detector_c_lib",
        "//mediapipe/tasks/c/vision/pose_landmarker:pose_landmarker_c_lib",
    ],
)
