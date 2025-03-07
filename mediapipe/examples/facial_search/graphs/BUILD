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

load("@rules_cc//cc:defs.bzl", "cc_library")
load("//mediapipe/framework/tool:mediapipe_graph.bzl", "mediapipe_binary_graph", "mediapipe_simple_subgraph")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cpu_calculators",
    deps = [
        ":subgraph_face_embeddings_cpu",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:packet_presence_calculator",
        "//mediapipe/examples/facial_search/calculators:closest_embeddings_calculator",
        "//mediapipe/modules/face_detection:face_detection_short_range_cpu",
    ],
)

cc_library(
    name = "gpu_calculators",
    deps = [
        ":subgraph_face_embeddings_gpu",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:packet_presence_calculator",
        "//mediapipe/examples/facial_search/calculators:closest_embeddings_calculator",
        "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
        "//mediapipe/modules/face_detection:face_detection_short_range_gpu",
    ],
)

mediapipe_binary_graph(
    name = "facial_search_cpu_binary_graph",
    graph = "facial_search_cpu.pbtxt",
    output_name = "facial_search_cpu.binarypb",
    deps = [":cpu_calculators"],
)

mediapipe_binary_graph(
    name = "facial_search_gpu_binary_graph",
    graph = "facial_search_gpu.pbtxt",
    output_name = "facial_search_gpu.binarypb",
    deps = [":gpu_calculators"],
)

cc_library(
    name = "face_embeddings_calculators",
    deps = [
        "//mediapipe/calculators/image:image_cropping_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_floats_calculator",
        "//mediapipe/calculators/util:detections_to_rects_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "subgraph_face_embeddings_cpu",
    graph = "subgraph_face_embeddings_cpu.pbtxt",
    register_as = "FaceEmbeddingsSubgraph",
    deps = [":face_embeddings_calculators"],
)

mediapipe_simple_subgraph(
    name = "subgraph_face_embeddings_gpu",
    graph = "subgraph_face_embeddings_gpu.pbtxt",
    register_as = "FaceEmbeddingsSubgraph",
    deps = [
        ":face_embeddings_calculators",
        "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
    ],
)
