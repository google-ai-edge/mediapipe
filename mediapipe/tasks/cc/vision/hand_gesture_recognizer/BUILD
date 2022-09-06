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
    "//mediapipe/tasks:internal",
])

licenses(["notice"])

cc_library(
    name = "handedness_util",
    srcs = ["handedness_util.cc"],
    hdrs = ["handedness_util.h"],
    deps = [
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "handedness_util_test",
    srcs = ["handedness_util_test.cc"],
    deps = [
        ":handedness_util",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/port:gtest_main",
    ],
)

cc_library(
    name = "hand_gesture_recognizer_subgraph",
    srcs = ["hand_gesture_recognizer_subgraph.cc"],
    deps = [
        "//mediapipe/calculators/core:concatenate_vector_calculator",
        "//mediapipe/calculators/tensor:tensor_converter_calculator",
        "//mediapipe/calculators/tensor:tensors_to_classification_calculator_cc_proto",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components:classification_postprocessing",
        "//mediapipe/tasks/cc/components:classification_postprocessing_options_cc_proto",
        "//mediapipe/tasks/cc/components:image_preprocessing",
        "//mediapipe/tasks/cc/components/containers:classifications_cc_proto",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/core:model_task_graph",
        "//mediapipe/tasks/cc/core:utils",
        "//mediapipe/tasks/cc/core/proto:inference_subgraph_cc_proto",
        "//mediapipe/tasks/cc/vision/hand_gesture_recognizer/calculators:hand_landmarks_to_matrix_calculator",
        "//mediapipe/tasks/cc/vision/hand_gesture_recognizer/calculators:handedness_to_matrix_calculator",
        "//mediapipe/tasks/cc/vision/hand_gesture_recognizer/proto:hand_gesture_recognizer_subgraph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/hand_landmark:hand_landmark_detector_graph",
        "//mediapipe/tasks/cc/vision/utils:image_tensor_specs",
        "//mediapipe/tasks/metadata:metadata_schema_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)
