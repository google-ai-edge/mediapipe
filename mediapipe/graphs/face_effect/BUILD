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
    name = "face_effect_gpu_deps",
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:immediate_mux_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/framework/tool:switch_container",
        "//mediapipe/graphs/face_effect/subgraphs:single_face_geometry_from_detection_gpu",
        "//mediapipe/graphs/face_effect/subgraphs:single_face_geometry_from_landmarks_gpu",
        "//mediapipe/modules/face_geometry:effect_renderer_calculator",
        "//mediapipe/modules/face_geometry:env_generator_calculator",
    ],
)

mediapipe_binary_graph(
    name = "face_effect_gpu_binary_graph",
    graph = "face_effect_gpu.pbtxt",
    output_name = "face_effect_gpu.binarypb",
    deps = [":face_effect_gpu_deps"],
)
