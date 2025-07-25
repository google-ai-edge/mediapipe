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

# Docs for Mediapipe Tasks Hand Landmarker
# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
cc_library(
    name = "hand_landmarker",
    srcs = ["hand_landmarker.cc"],
    hdrs = ["hand_landmarker.h"],
    deps = [
        ":hand_landmarker_graph",
        ":hand_landmarker_result",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/processors:classifier_options",
        "//mediapipe/tasks/cc/components/processors/proto:classifier_options_cc_proto",
        "//mediapipe/tasks/cc/core:base_options",
        "//mediapipe/tasks/cc/core:base_task_api",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/core:task_runner",
        "//mediapipe/tasks/cc/core:utils",
        "//mediapipe/tasks/cc/core/proto:inference_subgraph_cc_proto",
        "//mediapipe/tasks/cc/vision/core:base_vision_task_api",
        "//mediapipe/tasks/cc/vision/core:image_processing_options",
        "//mediapipe/tasks/cc/vision/core:running_mode",
        "//mediapipe/tasks/cc/vision/core:vision_task_api_factory",
        "//mediapipe/tasks/cc/vision/hand_detector/proto:hand_detector_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_landmarker_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_landmarks_detector_graph_options_cc_proto",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "hand_landmarks_detector_graph",
    srcs = ["hand_landmarks_detector_graph.cc"],
    deps = [
        "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_landmarks_detector_graph_options_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/core:split_vector_calculator_cc_proto",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/tensor:tensors_to_classification_calculator",
        "//mediapipe/calculators/tensor:tensors_to_classification_calculator_cc_proto",
        "//mediapipe/calculators/tensor:tensors_to_floats_calculator",
        "//mediapipe/calculators/tensor:tensors_to_landmarks_calculator",
        "//mediapipe/calculators/tensor:tensors_to_landmarks_calculator_cc_proto",
        "//mediapipe/calculators/util:landmark_letterbox_removal_calculator",
        "//mediapipe/calculators/util:landmark_projection_calculator",
        "//mediapipe/calculators/util:rect_transformation_calculator",
        "//mediapipe/calculators/util:rect_transformation_calculator_cc_proto",
        "//mediapipe/calculators/util:thresholding_calculator",
        "//mediapipe/calculators/util:thresholding_calculator_cc_proto",
        "//mediapipe/calculators/util:world_landmark_projection_calculator",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:tensor",
        # TODO: move calculators in modules/hand_landmark/calculators to tasks dir.
        "//mediapipe/modules/hand_landmark/calculators:hand_landmarks_to_rect_calculator",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/utils:gate",
        "//mediapipe/tasks/cc/components/processors:image_preprocessing_graph",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/core:model_task_graph",
        "//mediapipe/tasks/cc/core:utils",
        "//mediapipe/tasks/cc/core/proto:inference_subgraph_cc_proto",
        "//mediapipe/tasks/cc/vision/utils:image_tensor_specs",
        "//mediapipe/util:label_map_cc_proto",
        "//mediapipe/util:label_map_util",
        "//mediapipe/tasks/cc/metadata:metadata_extractor",
        "//mediapipe/tasks/metadata:metadata_schema_cc",
    ],
    alwayslink = 1,
)

# TODO: Enable this test

cc_library(
    name = "hand_landmarker_graph",
    srcs = ["hand_landmarker_graph.cc"],
    deps = [
        ":hand_landmarks_detector_graph",
        "//mediapipe/calculators/core:begin_loop_calculator",
        "//mediapipe/calculators/core:clip_vector_size_calculator_cc_proto",
        "//mediapipe/calculators/core:end_loop_calculator",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:gate_calculator_cc_proto",
        "//mediapipe/calculators/core:pass_through_calculator",
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/util:collection_has_min_size_calculator",
        "//mediapipe/calculators/util:collection_has_min_size_calculator_cc_proto",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/utils:gate",
        "//mediapipe/tasks/cc/core:model_asset_bundle_resources",
        "//mediapipe/tasks/cc/core:model_resources_cache",
        "//mediapipe/tasks/cc/core:model_task_graph",
        "//mediapipe/tasks/cc/core:utils",
        "//mediapipe/tasks/cc/metadata/utils:zip_utils",
        "//mediapipe/tasks/cc/vision/hand_detector:hand_detector_graph",
        "//mediapipe/tasks/cc/vision/hand_detector/proto:hand_detector_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/hand_landmarker/calculators:hand_association_calculator",
        "//mediapipe/tasks/cc/vision/hand_landmarker/calculators:hand_association_calculator_cc_proto",
        "//mediapipe/tasks/cc/vision/hand_landmarker/calculators:hand_landmarks_deduplication_calculator",
        "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_landmarker_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_landmarks_detector_graph_options_cc_proto",
        "//mediapipe/util:graph_builder_utils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "hand_landmarks_connections",
    hdrs = ["hand_landmarks_connections.h"],
)

# TODO: open source hand joints graph

cc_library(
    name = "hand_roi_refinement_graph",
    srcs = ["hand_roi_refinement_graph.cc"],
    deps = [
        "//mediapipe/calculators/tensor:image_to_tensor_calculator_cc_proto",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/tensor:tensors_to_landmarks_calculator",
        "//mediapipe/calculators/tensor:tensors_to_landmarks_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2/stream:detections_to_rects",
        "//mediapipe/framework/api2/stream:landmarks_projection",
        "//mediapipe/framework/api2/stream:landmarks_to_detection",
        "//mediapipe/framework/api2/stream:rect_transformation",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "//mediapipe/tasks/cc/components/processors:image_preprocessing_graph",
        "//mediapipe/tasks/cc/components/processors/proto:image_preprocessing_graph_options_cc_proto",
        "//mediapipe/tasks/cc/core:model_task_graph",
        "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_roi_refinement_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/utils:image_tensor_specs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

cc_library(
    name = "hand_landmarker_result",
    srcs = ["hand_landmarker_result.cc"],
    hdrs = ["hand_landmarker_result.h"],
    deps = [
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/tasks/cc/components/containers:classification_result",
        "//mediapipe/tasks/cc/components/containers:landmark",
    ],
)

# TODO: Enable this test

cc_library(
    name = "hand_topology",
    hdrs = ["hand_topology.h"],
)
