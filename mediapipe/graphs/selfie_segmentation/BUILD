# Copyright 2021 The MediaPipe Authors.
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
load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_binary_graph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "selfie_segmentation_gpu_deps",
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/image:recolor_calculator",
        "//mediapipe/modules/selfie_segmentation:selfie_segmentation_gpu",
    ],
)

mediapipe_binary_graph(
    name = "selfie_segmentation_gpu_binary_graph",
    graph = "selfie_segmentation_gpu.pbtxt",
    output_name = "selfie_segmentation_gpu.binarypb",
    deps = [":selfie_segmentation_gpu_deps"],
)

cc_library(
    name = "selfie_segmentation_cpu_deps",
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/image:recolor_calculator",
        "//mediapipe/modules/selfie_segmentation:selfie_segmentation_cpu",
    ],
)

mediapipe_binary_graph(
    name = "selfie_segmentation_cpu_binary_graph",
    graph = "selfie_segmentation_cpu.pbtxt",
    output_name = "selfie_segmentation_cpu.binarypb",
    deps = [":selfie_segmentation_cpu_deps"],
)
