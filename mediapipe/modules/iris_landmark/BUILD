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

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

mediapipe_simple_subgraph(
    name = "iris_landmark_cpu",
    graph = "iris_landmark_cpu.pbtxt",
    register_as = "IrisLandmarkCpu",
    deps = [
        "//mediapipe/calculators/core:clip_vector_size_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/image:image_cropping_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_floats_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_landmarks_calculator",
        "//mediapipe/calculators/util:landmark_letterbox_removal_calculator",
        "//mediapipe/calculators/util:landmark_projection_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "iris_landmark_gpu",
    graph = "iris_landmark_gpu.pbtxt",
    register_as = "IrisLandmarkGpu",
    deps = [
        "//mediapipe/calculators/core:clip_vector_size_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/image:image_cropping_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_floats_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_landmarks_calculator",
        "//mediapipe/calculators/util:landmark_letterbox_removal_calculator",
        "//mediapipe/calculators/util:landmark_projection_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "iris_landmark_left_and_right_gpu",
    graph = "iris_landmark_left_and_right_gpu.pbtxt",
    register_as = "IrisLandmarkLeftAndRightGpu",
    deps = [
        ":iris_landmark_gpu",
        ":iris_landmark_landmarks_to_roi",
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:side_packet_to_stream_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "iris_landmark_left_and_right_cpu",
    graph = "iris_landmark_left_and_right_cpu.pbtxt",
    register_as = "IrisLandmarkLeftAndRightCpu",
    deps = [
        ":iris_landmark_cpu",
        ":iris_landmark_landmarks_to_roi",
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:side_packet_to_stream_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
    ],
)

exports_files(
    srcs = [
        "iris_landmark.tflite",
    ],
)

mediapipe_simple_subgraph(
    name = "iris_landmark_landmarks_to_roi",
    graph = "iris_landmark_landmarks_to_roi.pbtxt",
    register_as = "IrisLandmarkLandmarksToRoi",
    deps = [
        "//mediapipe/calculators/util:detections_to_rects_calculator",
        "//mediapipe/calculators/util:landmarks_to_detection_calculator",
        "//mediapipe/calculators/util:rect_transformation_calculator",
    ],
)
