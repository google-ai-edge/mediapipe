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
    "mediapipe_binary_graph",
)

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "iris_depth_cpu_deps",
    deps = [
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/image:image_file_properties_calculator",
        "//mediapipe/calculators/image:opencv_encoded_image_to_image_frame_calculator",
        "//mediapipe/calculators/image:opencv_image_encoder_calculator",
        "//mediapipe/graphs/iris_tracking/subgraphs:iris_and_depth_renderer_cpu",
        "//mediapipe/modules/face_landmark:face_landmark_front_cpu",
        "//mediapipe/modules/iris_landmark:iris_landmark_left_and_right_cpu",
    ],
)

cc_library(
    name = "iris_tracking_cpu_deps",
    deps = [
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/graphs/iris_tracking/subgraphs:iris_renderer_cpu",
        "//mediapipe/modules/face_landmark:face_landmark_front_cpu",
        "//mediapipe/modules/iris_landmark:iris_landmark_left_and_right_cpu",
    ],
)

cc_library(
    name = "iris_tracking_cpu_video_input_deps",
    deps = [
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/video:opencv_video_decoder_calculator",
        "//mediapipe/calculators/video:opencv_video_encoder_calculator",
        "//mediapipe/graphs/iris_tracking/subgraphs:iris_renderer_cpu",
        "//mediapipe/modules/face_landmark:face_landmark_front_cpu",
        "//mediapipe/modules/iris_landmark:iris_landmark_left_and_right_cpu",
    ],
)

cc_library(
    name = "iris_tracking_gpu_deps",
    deps = [
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/graphs/iris_tracking/subgraphs:iris_and_depth_renderer_gpu",
        "//mediapipe/modules/face_landmark:face_landmark_front_gpu",
        "//mediapipe/modules/iris_landmark:iris_landmark_left_and_right_gpu",
    ],
)

mediapipe_binary_graph(
    name = "iris_tracking_gpu_binary_graph",
    graph = "iris_tracking_gpu.pbtxt",
    output_name = "iris_tracking_gpu.binarypb",
    deps = [":iris_tracking_gpu_deps"],
)
