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
    name = "pose_landmark_upper_body_by_roi_gpu",
    graph = "pose_landmark_upper_body_by_roi_gpu.pbtxt",
    register_as = "PoseLandmarkUpperBodyByRoiGpu",
    deps = [
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/image:image_cropping_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_floats_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_landmarks_calculator",
        "//mediapipe/calculators/util:landmark_letterbox_removal_calculator",
        "//mediapipe/calculators/util:landmark_projection_calculator",
        "//mediapipe/calculators/util:thresholding_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "pose_landmark_upper_body_by_roi_cpu",
    graph = "pose_landmark_upper_body_by_roi_cpu.pbtxt",
    register_as = "PoseLandmarkUpperBodyByRoiCpu",
    deps = [
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/image:image_cropping_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_floats_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_landmarks_calculator",
        "//mediapipe/calculators/util:landmark_letterbox_removal_calculator",
        "//mediapipe/calculators/util:landmark_projection_calculator",
        "//mediapipe/calculators/util:thresholding_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "pose_landmark_upper_body_gpu",
    graph = "pose_landmark_upper_body_gpu.pbtxt",
    register_as = "PoseLandmarkUpperBodyGpu",
    deps = [
        ":pose_detection_to_roi",
        ":pose_landmark_upper_body_by_roi_gpu",
        ":pose_landmark_upper_body_landmarks_to_roi",
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
    name = "pose_landmark_upper_body_cpu",
    graph = "pose_landmark_upper_body_cpu.pbtxt",
    register_as = "PoseLandmarkUpperBodyCpu",
    deps = [
        ":pose_detection_to_roi",
        ":pose_landmark_upper_body_by_roi_cpu",
        ":pose_landmark_upper_body_landmarks_to_roi",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:merge_calculator",
        "//mediapipe/calculators/core:packet_presence_calculator",
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/modules/pose_detection:pose_detection_cpu",
    ],
)

exports_files(
    srcs = [
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
    ],
)

mediapipe_simple_subgraph(
    name = "pose_landmark_upper_body_landmarks_to_roi",
    graph = "pose_landmark_upper_body_landmarks_to_roi.pbtxt",
    register_as = "PoseLandmarkUpperBodyLandmarksToRoi",
    deps = [
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/util:alignment_points_to_rects_calculator",
        "//mediapipe/calculators/util:landmarks_to_detection_calculator",
        "//mediapipe/calculators/util:rect_transformation_calculator",
    ],
)
