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

exports_files(glob([
    "*.pbtxt",
]))

cc_library(
    name = "mobile_calculators",
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/calculators/core:packet_resampler_calculator",
        "//mediapipe/calculators/image:image_cropping_calculator",
        "//mediapipe/gpu:gl_scaler_calculator",
        "//mediapipe/graphs/object_detection_3d/calculators:annotations_to_model_matrices_calculator",
        "//mediapipe/graphs/object_detection_3d/calculators:gl_animation_overlay_calculator",
        "//mediapipe/graphs/object_detection_3d/subgraphs:objectron_detection_gpu",
        "//mediapipe/graphs/object_detection_3d/subgraphs:objectron_tracking_gpu",
    ],
)

mediapipe_binary_graph(
    name = "mobile_gpu_binary_graph_shoe",
    graph = "shoe_classic_occlusion_tracking.pbtxt",
    output_name = "mobile_gpu_shoe.binarypb",
    visibility = ["//visibility:public"],
    deps = [":mobile_calculators"],
)

mediapipe_binary_graph(
    name = "mobile_gpu_binary_graph_chair",
    graph = "chair_classic_occlusion_tracking.pbtxt",
    output_name = "mobile_gpu_chair.binarypb",
    visibility = ["//visibility:public"],
    deps = [":mobile_calculators"],
)
