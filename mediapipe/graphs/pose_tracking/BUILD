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

cc_library(
    name = "upper_body_pose_tracking_gpu_deps",
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/util:landmarks_smoothing_calculator",
        "//mediapipe/graphs/pose_tracking/subgraphs:upper_body_pose_renderer_gpu",
        "//mediapipe/modules/pose_landmark:pose_landmark_upper_body_gpu",
    ],
)

mediapipe_binary_graph(
    name = "upper_body_pose_tracking_gpu_binary_graph",
    graph = "upper_body_pose_tracking_gpu.pbtxt",
    output_name = "upper_body_pose_tracking_gpu.binarypb",
    deps = [":upper_body_pose_tracking_gpu_deps"],
)

cc_library(
    name = "upper_body_pose_tracking_cpu_deps",
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/util:landmarks_smoothing_calculator",
        "//mediapipe/graphs/pose_tracking/subgraphs:upper_body_pose_renderer_cpu",
        "//mediapipe/modules/pose_landmark:pose_landmark_upper_body_cpu",
    ],
)

mediapipe_binary_graph(
    name = "upper_body_pose_tracking_cpu_binary_graph",
    graph = "upper_body_pose_tracking_cpu.pbtxt",
    output_name = "upper_body_pose_tracking_cpu.binarypb",
    deps = [":upper_body_pose_tracking_cpu_deps"],
)
