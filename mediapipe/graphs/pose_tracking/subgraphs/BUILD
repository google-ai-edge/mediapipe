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

mediapipe_simple_subgraph(
    name = "pose_renderer_gpu",
    graph = "pose_renderer_gpu.pbtxt",
    register_as = "PoseRendererGpu",
    deps = [
        "//mediapipe/calculators/core:split_proto_list_calculator",
        "//mediapipe/calculators/image:recolor_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:detections_to_render_data_calculator",
        "//mediapipe/calculators/util:landmarks_to_render_data_calculator",
        "//mediapipe/calculators/util:rect_to_render_data_calculator",
        "//mediapipe/calculators/util:rect_to_render_scale_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "pose_renderer_cpu",
    graph = "pose_renderer_cpu.pbtxt",
    register_as = "PoseRendererCpu",
    deps = [
        "//mediapipe/calculators/core:split_proto_list_calculator",
        "//mediapipe/calculators/image:recolor_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:detections_to_render_data_calculator",
        "//mediapipe/calculators/util:landmarks_to_render_data_calculator",
        "//mediapipe/calculators/util:rect_to_render_data_calculator",
        "//mediapipe/calculators/util:rect_to_render_scale_calculator",
    ],
)
