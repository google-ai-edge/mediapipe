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
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")
load("//mediapipe/framework:mediapipe_cc_test.bzl", "mediapipe_cc_test")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_simple_subgraph(
    name = "face_detection_short_range_by_roi_cpu",
    graph = "face_detection_short_range_by_roi_cpu.pbtxt",
    register_as = "FaceDetectionShortRangeByRoiCpu",
    deps = [
        ":face_detection_short_range",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/util:to_image_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "face_detection_short_range_by_roi_gpu",
    graph = "face_detection_short_range_by_roi_gpu.pbtxt",
    register_as = "FaceDetectionShortRangeByRoiGpu",
    deps = [
        ":face_detection_short_range",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/util:to_image_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "face_detection_short_range_cpu",
    graph = "face_detection_short_range_cpu.pbtxt",
    register_as = "FaceDetectionShortRangeCpu",
    deps = [
        ":face_detection_short_range",
    ],
)

mediapipe_simple_subgraph(
    name = "face_detection_short_range_gpu",
    graph = "face_detection_short_range_gpu.pbtxt",
    register_as = "FaceDetectionShortRangeGpu",
    deps = [
        ":face_detection_short_range",
    ],
)

mediapipe_simple_subgraph(
    name = "face_detection_short_range",
    graph = "face_detection_short_range.pbtxt",
    register_as = "FaceDetectionShortRange",
    deps = [
        ":face_detection",
    ],
)

mediapipe_simple_subgraph(
    name = "face_detection_full_range",
    graph = "face_detection_full_range.pbtxt",
    register_as = "FaceDetectionFullRange",
    deps = [
        ":face_detection",
    ],
)

mediapipe_simple_subgraph(
    name = "face_detection_without_roi",
    graph = "face_detection_without_roi.pbtxt",
    register_as = "FaceDetectionWithoutRoi",
    deps = [
        ":face_detection",
    ],
)

mediapipe_simple_subgraph(
    name = "face_detection",
    graph = "face_detection.pbtxt",
    register_as = "FaceDetection",
    deps = [
        ":face_detection_cc_proto",
        ":face_detection_options_lib",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/tensor:tensors_to_detections_calculator",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "//mediapipe/calculators/util:detection_projection_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/calculators/util:to_image_calculator",
    ],
)

mediapipe_proto_library(
    name = "face_detection_proto",
    srcs = ["face_detection.proto"],
    deps = [
        "//mediapipe/calculators/core:gate_calculator_proto",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator_proto",
        "//mediapipe/calculators/tensor:inference_calculator_proto",
        "//mediapipe/calculators/tensor:tensors_to_detections_calculator_proto",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator_proto",
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/gpu:gpu_origin_proto",
    ],
)

mediapipe_simple_subgraph(
    name = "face_detection_full_range_cpu",
    graph = "face_detection_full_range_cpu.pbtxt",
    register_as = "FaceDetectionFullRangeCpu",
    deps = [
        ":face_detection_full_range",
    ],
)

mediapipe_simple_subgraph(
    name = "face_detection_full_range_gpu",
    graph = "face_detection_full_range_gpu.pbtxt",
    register_as = "FaceDetectionFullRangeGpu",
    deps = [
        ":face_detection_full_range",
    ],
)

mediapipe_simple_subgraph(
    name = "face_detection_short_range_image",
    graph = "face_detection_short_range_image.pbtxt",
    register_as = "FaceDetectionShortRangeImage",
    deps = [
        ":face_detection_short_range",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "face_detection_full_range_image",
    graph = "face_detection_full_range_image.pbtxt",
    register_as = "FaceDetectionFullRangeImage",
    deps = [
        ":face_detection_full_range",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
    ],
)

exports_files(
    srcs = [
        "face_detection_full_range.tflite",
        "face_detection_full_range_sparse.tflite",
        "face_detection_short_range.tflite",
    ],
)
