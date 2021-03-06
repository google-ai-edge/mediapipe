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
    name = "face_detection_front_by_roi_cpu",
    graph = "face_detection_front_by_roi_cpu.pbtxt",
    register_as = "FaceDetectionFrontByRoiCpu",
    deps = [
        ":face_detection_front_common",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/util:to_image_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "face_detection_front_by_roi_gpu",
    graph = "face_detection_front_by_roi_gpu.pbtxt",
    register_as = "FaceDetectionFrontByRoiGpu",
    deps = [
        ":face_detection_front_common",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/util:to_image_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "face_detection_front_cpu",
    graph = "face_detection_front_cpu.pbtxt",
    register_as = "FaceDetectionFrontCpu",
    deps = [
        ":face_detection_front_common",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/util:to_image_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "face_detection_front_gpu",
    graph = "face_detection_front_gpu.pbtxt",
    register_as = "FaceDetectionFrontGpu",
    deps = [
        ":face_detection_front_common",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/util:to_image_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "face_detection_front_common",
    graph = "face_detection_front_common.pbtxt",
    register_as = "FaceDetectionFrontCommon",
    deps = [
        "//mediapipe/calculators/tensor:tensors_to_detections_calculator",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "//mediapipe/calculators/util:detection_projection_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
    ],
)

exports_files(
    srcs = [
        "face_detection_front.tflite",
    ],
)
