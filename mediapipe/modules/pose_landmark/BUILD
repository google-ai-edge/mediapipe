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
    "mediapipe_simple_subgraph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_simple_subgraph(
    name = "pose_landmark_model_loader",
    graph = "pose_landmark_model_loader.pbtxt",
    register_as = "PoseLandmarkModelLoader",
    deps = [
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/tflite:tflite_model_calculator",
        "//mediapipe/calculators/util:local_file_contents_calculator",
        "//mediapipe/framework/tool:switch_container",
    ],
)

mediapipe_simple_subgraph(
    name = "pose_landmark_by_roi_gpu",
    graph = "pose_landmark_by_roi_gpu.pbtxt",
    register_as = "PoseLandmarkByRoiGpu",
    deps = [
        ":pose_landmark_model_loader",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:split_normalized_landmark_list_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/tensor:tensors_to_floats_calculator",
        "//mediapipe/calculators/tensor:tensors_to_landmarks_calculator",
        "//mediapipe/calculators/util:landmark_letterbox_removal_calculator",
        "//mediapipe/calculators/util:landmark_projection_calculator",
        "//mediapipe/calculators/util:thresholding_calculator",
        "//mediapipe/framework/tool:switch_container",
    ],
)

mediapipe_simple_subgraph(
    name = "pose_landmark_by_roi_cpu",
    graph = "pose_landmark_by_roi_cpu.pbtxt",
    register_as = "PoseLandmarkByRoiCpu",
    deps = [
        ":pose_landmark_model_loader",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:split_normalized_landmark_list_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/tensor:tensors_to_floats_calculator",
        "//mediapipe/calculators/tensor:tensors_to_landmarks_calculator",
        "//mediapipe/calculators/util:landmark_letterbox_removal_calculator",
        "//mediapipe/calculators/util:landmark_projection_calculator",
        "//mediapipe/calculators/util:thresholding_calculator",
        "//mediapipe/framework/tool:switch_container",
    ],
)

mediapipe_simple_subgraph(
    name = "pose_landmark_filtering",
    graph = "pose_landmark_filtering.pbtxt",
    register_as = "PoseLandmarkFiltering",
    deps = [
        "//mediapipe/calculators/util:landmarks_smoothing_calculator",
        "//mediapipe/framework/tool:switch_container",
    ],
)

mediapipe_simple_subgraph(
    name = "pose_landmark_gpu",
    graph = "pose_landmark_gpu.pbtxt",
    register_as = "PoseLandmarkGpu",
    deps = [
        ":pose_detection_to_roi",
        ":pose_landmark_by_roi_gpu",
        ":pose_landmark_filtering",
        ":pose_landmarks_to_roi",
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:merge_calculator",
        "//mediapipe/calculators/core:packet_presence_calculator",
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/modules/pose_detection:pose_detection_gpu",
    ],
)

mediapipe_simple_subgraph(
    name = "pose_landmark_cpu",
    graph = "pose_landmark_cpu.pbtxt",
    register_as = "PoseLandmarkCpu",
    deps = [
        ":pose_detection_to_roi",
        ":pose_landmark_by_roi_cpu",
        ":pose_landmark_filtering",
        ":pose_landmarks_to_roi",
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:merge_calculator",
        "//mediapipe/calculators/core:packet_presence_calculator",
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/modules/pose_detection:pose_detection_cpu",
    ],
)

mediapipe_simple_subgraph(
    name = "pose_landmark_upper_body_gpu",
    graph = "pose_landmark_upper_body_gpu.pbtxt",
    register_as = "PoseLandmarkUpperBodyGpu",
    deps = [
        ":pose_landmark_gpu",
        "//mediapipe/calculators/core:constant_side_packet_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "pose_landmark_upper_body_cpu",
    graph = "pose_landmark_upper_body_cpu.pbtxt",
    register_as = "PoseLandmarkUpperBodyCpu",
    deps = [
        ":pose_landmark_cpu",
        "//mediapipe/calculators/core:constant_side_packet_calculator",
    ],
)

exports_files(
    srcs = [
        "pose_landmark_full_body.tflite",
        "pose_landmark_upper_body.tflite",
    ],
)

mediapipe_simple_subgraph(
    name = "pose_detection_to_roi",
    graph = "pose_detection_to_roi.pbtxt",
    register_as = "PoseDetectionToRoi",
    deps = [
        "//mediapipe/calculators/util:alignment_points_to_rects_calculator",
        "//mediapipe/calculators/util:rect_transformation_calculator",
        "//mediapipe/framework/tool:switch_container",
    ],
)

mediapipe_simple_subgraph(
    name = "pose_landmarks_to_roi",
    graph = "pose_landmarks_to_roi.pbtxt",
    register_as = "PoseLandmarksToRoi",
    deps = [
        "//mediapipe/calculators/util:alignment_points_to_rects_calculator",
        "//mediapipe/calculators/util:landmarks_to_detection_calculator",
        "//mediapipe/calculators/util:rect_transformation_calculator",
    ],
)
