# Copyright 2023 The MediaPipe Authors.
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
load("@org_tensorflow//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(default_visibility = [
    "//mediapipe/tasks:internal",
])

licenses(["notice"])

cc_library_with_tflite(
    name = "face_detector_graph",
    srcs = ["face_detector_graph.cc"],
    tflite_deps = [
        "//mediapipe/tasks/cc/components/processors:image_preprocessing_graph",
        "//mediapipe/tasks/cc/core:model_task_graph",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/vision/utils:image_tensor_specs",
    ],
    deps = [
        "//mediapipe/calculators/core:clip_vector_size_calculator",
        "//mediapipe/calculators/core:clip_vector_size_calculator_cc_proto",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator_cc_proto",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/tensor:tensors_to_detections_calculator",
        "//mediapipe/calculators/tensor:tensors_to_detections_calculator_cc_proto",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator_cc_proto",
        "//mediapipe/calculators/util:detection_projection_calculator",
        "//mediapipe/calculators/util:detection_transformation_calculator",
        "//mediapipe/calculators/util:detections_to_rects_calculator",
        "//mediapipe/calculators/util:detections_to_rects_calculator_cc_proto",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator_cc_proto",
        "//mediapipe/calculators/util:rect_transformation_calculator",
        "//mediapipe/calculators/util:rect_transformation_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/core:utils",
        "//mediapipe/tasks/cc/core/proto:inference_subgraph_cc_proto",
        "//mediapipe/tasks/cc/vision/face_detector/proto:face_detector_graph_options_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

cc_library_with_tflite(
    name = "face_detector",
    srcs = ["face_detector.cc"],
    hdrs = ["face_detector.h"],
    tflite_deps = [
        ":face_detector_graph",
        "//mediapipe/tasks/cc/vision/core:base_vision_task_api",
        "//mediapipe/tasks/cc/vision/core:vision_task_api_factory",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/tasks/cc/components/containers:detection_result",
        "//mediapipe/tasks/cc/core:base_options",
        "//mediapipe/tasks/cc/core:utils",
        "//mediapipe/tasks/cc/vision/core:image_processing_options",
        "//mediapipe/tasks/cc/vision/core:running_mode",
        "//mediapipe/tasks/cc/vision/face_detector/proto:face_detector_graph_options_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)
