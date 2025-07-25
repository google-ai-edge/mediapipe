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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "hand_detector_graph",
    srcs = ["hand_detector_graph.cc"],
    deps = [
        "//mediapipe/calculators/core:clip_vector_size_calculator",
        "//mediapipe/calculators/core:clip_vector_size_calculator_cc_proto",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator_cc_proto",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/tensor:tensors_to_detections_calculator",
        "//mediapipe/calculators/tensor:tensors_to_detections_calculator_cc_proto",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator_cc_proto",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator_cc_proto",
        "//mediapipe/calculators/util:detection_projection_calculator",
        "//mediapipe/calculators/util:detections_to_rects_calculator",
        "//mediapipe/calculators/util:detections_to_rects_calculator_cc_proto",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator_cc_proto",
        "//mediapipe/calculators/util:rect_transformation_calculator",
        "//mediapipe/calculators/util:rect_transformation_calculator_cc_proto",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/processors:image_preprocessing_graph",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/core:model_task_graph",
        "//mediapipe/tasks/cc/core:utils",
        "//mediapipe/tasks/cc/core/proto:inference_subgraph_cc_proto",
        "//mediapipe/tasks/cc/vision/hand_detector/proto:hand_detector_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/utils:image_tensor_specs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)
