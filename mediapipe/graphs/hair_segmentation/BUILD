# Copyright 2019 The MediaPipeOSS Authors.
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

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "android_calculators",
    deps = [
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/calculators/core:real_time_flow_limiter_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/image:recolor_calculator",
        "//mediapipe/calculators/image:set_alpha_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_custom_op_resolver_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_segmentation_calculator",
    ],
)

load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_binary_graph",
)

mediapipe_binary_graph(
    name = "android_gpu_binary_graph",
    graph = "hair_segmentation_android_gpu.pbtxt",
    output_name = "android_gpu.binarypb",
    deps = [
        "//mediapipe/calculators/image:image_transformation_calculator_proto",
        "//mediapipe/calculators/image:recolor_calculator_proto",
        "//mediapipe/calculators/image:set_alpha_calculator_proto",
        "//mediapipe/calculators/tflite:tflite_converter_calculator_proto",
        "//mediapipe/calculators/tflite:tflite_custom_op_resolver_calculator_proto",
        "//mediapipe/calculators/tflite:tflite_inference_calculator_proto",
        "//mediapipe/calculators/tflite:tflite_tensors_to_segmentation_calculator_proto",
    ],
)
