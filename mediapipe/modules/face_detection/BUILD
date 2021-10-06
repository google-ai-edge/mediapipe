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
    name = "face_detection_short_range_by_roi_cpu",
    graph = "face_detection_short_range_by_roi_cpu.pbtxt",
    register_as = "FaceDetectionShortRangeByRoiCpu",
    deps = [
        ":face_detection_short_range_common",
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
        ":face_detection_short_range_common",
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
        ":face_detection_short_range_common",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/util:to_image_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "face_detection_short_range_gpu",
    graph = "face_detection_short_range_gpu.pbtxt",
    register_as = "FaceDetectionShortRangeGpu",
    deps = [
        ":face_detection_short_range_common",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/util:to_image_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "face_detection_short_range_common",
    graph = "face_detection_short_range_common.pbtxt",
    register_as = "FaceDetectionShortRangeCommon",
    deps = [
        "//mediapipe/calculators/tensor:tensors_to_detections_calculator",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "//mediapipe/calculators/util:detection_projection_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "face_detection_full_range_cpu",
    graph = "face_detection_full_range_cpu.pbtxt",
    register_as = "FaceDetectionFullRangeCpu",
    deps = [
        ":face_detection_full_range_common",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/util:to_image_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "face_detection_full_range_gpu",
    graph = "face_detection_full_range_gpu.pbtxt",
    register_as = "FaceDetectionFullRangeGpu",
    deps = [
        ":face_detection_full_range_common",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/util:to_image_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "face_detection_full_range_common",
    graph = "face_detection_full_range_common.pbtxt",
    register_as = "FaceDetectionFullRangeCommon",
    deps = [
        "//mediapipe/calculators/tensor:tensors_to_detections_calculator",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "//mediapipe/calculators/util:detection_projection_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "face_detection_short_range_image",
    graph = "face_detection_short_range_image.pbtxt",
    register_as = "FaceDetectionShortRangeImage",
    deps = [
        ":face_detection_short_range_common",
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
        ":face_detection_full_range_common",
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
