# Copyright 2020 The MediaPipe Authors.
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

load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_binary_graph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

exports_files(glob([
    "*.pbtxt",
]))

cc_library(
    name = "mobile_calculators",
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:merge_calculator",
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/calculators/image:image_cropping_calculator",
        "//mediapipe/graphs/object_detection_3d/calculators:annotations_to_model_matrices_calculator",
        "//mediapipe/graphs/object_detection_3d/calculators:gl_animation_overlay_calculator",
        "//mediapipe/graphs/object_detection_3d/subgraphs:box_landmark_gpu",
        "//mediapipe/graphs/object_detection_3d/subgraphs:object_detection_oid_v4_gpu",
    ],
)

cc_library(
    name = "mobile_calculators_1stage",
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/calculators/core:packet_resampler_calculator",
        "//mediapipe/calculators/image:image_cropping_calculator",
        "//mediapipe/gpu:gl_scaler_calculator",
        "//mediapipe/graphs/object_detection_3d/calculators:annotations_to_model_matrices_calculator",
        "//mediapipe/graphs/object_detection_3d/calculators:gl_animation_overlay_calculator",
        "//mediapipe/graphs/object_detection_3d/subgraphs:objectron_detection_gpu",
        "//mediapipe/graphs/object_detection_3d/subgraphs:objectron_tracking_gpu",
    ],
)

mediapipe_binary_graph(
    name = "mobile_gpu_binary_graph",
    graph = "object_occlusion_tracking.pbtxt",
    output_name = "mobile_gpu_binary_graph.binarypb",
    visibility = ["//visibility:public"],
    deps = [":mobile_calculators"],
)

mediapipe_binary_graph(
    name = "mobile_gpu_1stage_binary_graph",
    graph = "object_occlusion_tracking_1stage.pbtxt",
    output_name = "mobile_gpu_1stage_binary_graph.binarypb",
    visibility = ["//visibility:public"],
    deps = [":mobile_calculators_1stage"],
)
