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
        "//mediapipe/calculators/core:concatenate_normalized_landmark_list_calculator",
        "//mediapipe/calculators/core:concatenate_vector_calculator",
        "//mediapipe/calculators/core:split_normalized_landmark_list_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "//mediapipe/calculators/util:detections_to_render_data_calculator",
        "//mediapipe/calculators/util:landmarks_to_render_data_calculator",
        "//mediapipe/calculators/util:rect_to_render_data_calculator",
        "//mediapipe/graphs/face_mesh/calculators:face_landmarks_to_render_data_calculator",
        "//mediapipe/graphs/iris_tracking/calculators:iris_to_render_data_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "iris_and_depth_renderer_gpu",
    graph = "iris_and_depth_renderer_gpu.pbtxt",
    register_as = "IrisAndDepthRendererGpu",
    deps = [
        ":renderer_calculators",
        "//mediapipe/graphs/iris_tracking/calculators:iris_to_depth_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "iris_renderer_cpu",
    graph = "iris_renderer_cpu.pbtxt",
    register_as = "IrisRendererCpu",
    deps = [
        ":renderer_calculators",
    ],
)

mediapipe_simple_subgraph(
    name = "iris_and_depth_renderer_cpu",
    graph = "iris_and_depth_renderer_cpu.pbtxt",
    register_as = "IrisAndDepthRendererCpu",
    deps = [
        ":renderer_calculators",
        "//mediapipe/graphs/iris_tracking/calculators:iris_to_depth_calculator",
    ],
)
