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

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mobile_calculators",
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:tensor_converter_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
         "//mediapipe/calculators/tensor:tensors_to_segmentation_calculator",
         "//mediapipe/calculators/util:to_image_calculator",
         "//mediapipe/calculators/util:from_image_calculator",
         "//mediapipe/calculators/image:image_properties_calculator",
         "//mediapipe/modules/face_landmark:face_landmark_front_gpu",
          "//mediapipe/calculators/image_style:fast_utils_calculator",
          "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
         "//mediapipe/calculators/core:constant_side_packet_calculator",
    ],
)

cc_library(
    name = "desktop_calculators",
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
            "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/tensor:tensor_converter_calculator",
         "//mediapipe/calculators/tensor:tensors_to_segmentation_calculator",
         "//mediapipe/calculators/util:to_image_calculator",
         "//mediapipe/calculators/util:from_image_calculator",
           "//mediapipe/modules/face_landmark:face_landmark_front_cpu",
           "//mediapipe/calculators/image_style:fast_utils_calculator",
           "//mediapipe/calculators/image:image_properties_calculator",
              "//mediapipe/calculators/core:constant_side_packet_calculator",
    ],
)

mediapipe_binary_graph(
    name = "mobile_gpu_binary_graph",
    graph = "image_style_gpu.pbtxt",
    output_name = "mobile_gpu.binarypb",
    deps = [":mobile_calculators"],
)
