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

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

# Docs for Mediapipe Tasks Object Detector
# https://developers.google.com/mediapipe/solutions/vision/object_detector
cc_library(
    name = "object_detector",
    srcs = ["object_detector.cc"],
    hdrs = ["object_detector.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":object_detector_graph",
        "//mediapipe/calculators/core:concatenate_vector_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/calculators:score_calibration_calculator",
        "//mediapipe/tasks/cc/components/containers:detection_result",
        "//mediapipe/tasks/cc/core:base_options",
        "//mediapipe/tasks/cc/core:utils",
        "//mediapipe/tasks/cc/core/proto:base_options_cc_proto",
        "//mediapipe/tasks/cc/core/proto:inference_subgraph_cc_proto",
        "//mediapipe/tasks/cc/vision/core:base_vision_task_api",
        "//mediapipe/tasks/cc/vision/core:image_processing_options",
        "//mediapipe/tasks/cc/vision/core:running_mode",
        "//mediapipe/tasks/cc/vision/core:vision_task_api_factory",
        "//mediapipe/tasks/cc/vision/object_detector/proto:object_detector_options_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/lite/core/api:op_resolver",
    ],
)

cc_library(
    name = "object_detector_graph",
    srcs = ["object_detector_graph.cc"],
    deps = [
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/util:detection_projection_calculator",
        "//mediapipe/calculators/util:detection_transformation_calculator",
        "//mediapipe/calculators/util:detections_deduplicate_calculator",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/processors:detection_postprocessing_graph",
        "//mediapipe/tasks/cc/components/processors:image_preprocessing_graph",
        "//mediapipe/tasks/cc/components/processors/proto:detection_postprocessing_graph_options_cc_proto",
        "//mediapipe/tasks/cc/components/processors/proto:detector_options_cc_proto",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/core:model_task_graph",
        "//mediapipe/tasks/cc/core/proto:inference_subgraph_cc_proto",
        "//mediapipe/tasks/cc/vision/object_detector/proto:object_detector_options_cc_proto",
        "//mediapipe/tasks/metadata:metadata_schema_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

# TODO: This test fails in OSS
