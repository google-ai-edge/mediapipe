# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl" ,
    "mediapipe_binary_graph" ,
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mobile_calculators" ,
    deps = [
    "//mediapipe/graphs/deformation/calculators:face_processor_calculator" ,
    "//mediapipe/graphs/deformation/calculators:warp_affine_calculator" ,
    "//mediapipe/calculators/core:flow_limiter_calculator" ,
    "//mediapipe/calculators/image:image_transformation_calculator" ,
    "//mediapipe/modules/face_landmark:face_landmark_front_gpu" ,
    "//mediapipe/calculators/core:constant_side_packet_calculator" ,
    "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator" ,
    "//mediapipe/gpu:image_frame_to_gpu_buffer_calculator" ,
    "//mediapipe/calculators/image:color_convert_calculator" ,
    "//mediapipe/calculators/image:image_properties_calculator" ,
    ],
)

cc_library(
    name = "desktop_calculators" ,
    deps = [
    "//mediapipe/graphs/deformation/calculators:face_processor_calculator" ,
    "//mediapipe/graphs/deformation/calculators:warp_affine_calculator" ,
    "//mediapipe/calculators/core:flow_limiter_calculator" ,
    "//mediapipe/calculators/image:image_transformation_calculator" ,
    "//mediapipe/modules/face_landmark:face_landmark_front_cpu" ,
    "//mediapipe/calculators/core:constant_side_packet_calculator" ,
    "//mediapipe/calculators/image:color_convert_calculator" ,
    "//mediapipe/calculators/image:image_properties_calculator" ,
    ],
)

mediapipe_binary_graph(
    name = "mobile_gpu_binary_graph" ,
    graph = "deformation_mobile.pbtxt" ,
    output_name = "mobile_gpu.binarypb" ,
    deps = [":mobile_calculators"],
)
