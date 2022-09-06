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

package(default_visibility = [
    "//mediapipe/app/xeno:__subpackages__",
    "//mediapipe/tasks:internal",
])

cc_library(
    name = "handedness_to_matrix_calculator",
    srcs = ["handedness_to_matrix_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/tasks/cc/vision/hand_gesture_recognizer:handedness_util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_test(
    name = "handedness_to_matrix_calculator_test",
    srcs = ["handedness_to_matrix_calculator_test.cc"],
    deps = [
        ":handedness_to_matrix_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "hand_landmarks_to_matrix_calculator",
    srcs = ["hand_landmarks_to_matrix_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_test(
    name = "hand_landmarks_to_matrix_calculator_test",
    srcs = ["hand_landmarks_to_matrix_calculator_test.cc"],
    deps = [
        ":hand_landmarks_to_matrix_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/memory",
    ],
)
