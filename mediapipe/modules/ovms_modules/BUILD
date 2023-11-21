# Copyright (c) 2023 Intel Corporation
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
#

load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_simple_subgraph",
)
load(
    "//mediapipe/framework/tool:mediapipe_files.bzl",
    "mediapipe_files",
)
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")
load("//mediapipe/framework:mediapipe_cc_test.bzl", "mediapipe_cc_test")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_simple_subgraph(
    name = "object_detection_ovms",
    graph = "object_detection_ovms.pbtxt",
    register_as = "ObjectDetectionOvms",
    deps = [
        "//mediapipe/graphs/object_detection:desktop_ovms_calculators",
        "@ovms//src:ovms_lib",
    ],
)

mediapipe_simple_subgraph(
    name = "holistic_tracking_ovms",
    graph = "holistic_tracking_ovms.pbtxt",
    register_as = "HolisticTrackingOvms",
    deps = [
        "//mediapipe/graphs/object_detection:desktop_ovms_calculators",
        "@ovms//src:ovms_lib",
    ],
)

mediapipe_simple_subgraph(
    name = "face_detection_ovms",
    graph = "face_detection_ovms.pbtxt",
    register_as = "FaceDetectionOvms",
    deps = [
        "//mediapipe/graphs/object_detection:desktop_ovms_calculators",
        "@ovms//src:ovms_lib",
    ],
)

mediapipe_simple_subgraph(
    name = "iris_tracking_ovms",
    graph = "iris_tracking_ovms.pbtxt",
    register_as = "IrisTrackingOvms",
    deps = [
        "//mediapipe/graphs/object_detection:desktop_ovms_calculators",
        "@ovms//src:ovms_lib",
    ],
)

mediapipe_simple_subgraph(
    name = "pose_landmarks_to_render_data",
    graph = "pose_landmarks_to_render_data.pbtxt",
    register_as = "PoseLandmarksToRenderData",
    deps = [
        "//mediapipe/calculators/core:concatenate_vector_calculator",
        "//mediapipe/calculators/core:split_proto_list_calculator",
        "//mediapipe/calculators/util:landmarks_to_render_data_calculator",
        "//mediapipe/calculators/util:rect_to_render_scale_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "pose_renderer_cpu",
    graph = "pose_renderer_cpu.pbtxt",
    register_as = "PoseRendererCpu",
    deps = [
        ":pose_landmarks_to_render_data",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/image:recolor_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:detections_to_render_data_calculator",
        "//mediapipe/calculators/util:rect_to_render_data_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "pose_tracking_ovms",
    graph = "pose_tracking_ovms.pbtxt",
    register_as = "PoseTrackingOvms",
    deps = [
        "//mediapipe/graphs/object_detection:desktop_ovms_calculators",
        ":pose_renderer_cpu",
        "@ovms//src:ovms_lib",
    ],
)
