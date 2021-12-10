# Copyright 2021 The MediaPipe Authors.
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
    name = "selfie_segmentation_model_loader",
    graph = "selfie_segmentation_model_loader.pbtxt",
    register_as = "SelfieSegmentationModelLoader",
    deps = [
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/tflite:tflite_model_calculator",
        "//mediapipe/calculators/util:local_file_contents_calculator",
        "//mediapipe/framework/tool:switch_container",
    ],
)

mediapipe_simple_subgraph(
    name = "selfie_segmentation_cpu",
    graph = "selfie_segmentation_cpu.pbtxt",
    register_as = "SelfieSegmentationCpu",
    deps = [
        ":selfie_segmentation_model_loader",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/tensor:tensors_to_segmentation_calculator",
        "//mediapipe/calculators/tflite:tflite_custom_op_resolver_calculator",
        "//mediapipe/calculators/util:from_image_calculator",
        "//mediapipe/framework/tool:switch_container",
    ],
)

mediapipe_simple_subgraph(
    name = "selfie_segmentation_gpu",
    graph = "selfie_segmentation_gpu.pbtxt",
    register_as = "SelfieSegmentationGpu",
    deps = [
        ":selfie_segmentation_model_loader",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/tensor:tensors_to_segmentation_calculator",
        "//mediapipe/calculators/tflite:tflite_custom_op_resolver_calculator",
        "//mediapipe/calculators/util:from_image_calculator",
        "//mediapipe/framework/tool:switch_container",
    ],
)

mediapipe_simple_subgraph(
    name = "selfie_segmentation_cpu_image",
    graph = "selfie_segmentation_cpu_image.pbtxt",
    register_as = "SelfieSegmentationCpuImage",
    deps = [
        ":selfie_segmentation_cpu",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/util:from_image_calculator",
        "//mediapipe/calculators/util:to_image_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "selfie_segmentation_gpu_image",
    graph = "selfie_segmentation_gpu_image.pbtxt",
    register_as = "SelfieSegmentationGpuImage",
    deps = [
        ":selfie_segmentation_gpu",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/util:from_image_calculator",
        "//mediapipe/calculators/util:to_image_calculator",
    ],
)

exports_files(
    srcs = [
        "selfie_segmentation.tflite",
        "selfie_segmentation_landscape.tflite",
    ],
)
