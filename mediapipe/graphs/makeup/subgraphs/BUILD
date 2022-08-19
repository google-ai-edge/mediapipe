# Copyright 2019 The MediaPipe Authors.
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
    "mediapipe_simple_subgraph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "renderer_calculators",
    deps = [
        "//mediapipe/calculators/beauty:smooth_face_calculator1",
        "//mediapipe/calculators/beauty:bilateral_filter_calculator",
        "//mediapipe/calculators/beauty:smooth_face_calculator2",
        "//mediapipe/calculators/beauty:draw_lipstick_calculator",
        "//mediapipe/calculators/beauty:whiten_teeth_calculator",
        "//mediapipe/calculators/beauty:merging_images_calculator",
        "//mediapipe/calculators/beauty:imageframe_to_mat_calculator",
        "//mediapipe/calculators/landmarks:landmarks_to_point_array_calculator",
        "//mediapipe/calculators/landmarks:points_to_face_box_calculator",
        "//mediapipe/calculators/landmarks:point_vector_to_mask_calculator",
        "//mediapipe/calculators/core:round_robin_demux_calculator",
        "//mediapipe/calculators/core:mux_calculator",    
    ],
)


mediapipe_simple_subgraph(
    name = "face_renderer_gpu",
    graph = "face_renderer_gpu.pbtxt",
    register_as = "FaceRendererGpu",
    deps = [
        ":renderer_calculators",
    ],
)


mediapipe_simple_subgraph(
    name = "face_renderer_cpu",
    graph = "face_renderer_cpu.pbtxt",
    register_as = "FaceRendererCpu",
    deps = [
        ":renderer_calculators",
    ],
)
