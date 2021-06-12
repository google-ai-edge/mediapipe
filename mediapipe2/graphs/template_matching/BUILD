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
    name = "template_matching_deps",
    deps = [
        "//mediapipe/calculators/image:feature_detector_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_floats_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:landmarks_to_render_data_calculator",
        "//mediapipe/calculators/util:timed_box_list_id_to_label_calculator",
        "//mediapipe/calculators/util:timed_box_list_to_render_data_calculator",
        "//mediapipe/calculators/video:box_detector_calculator",
    ],
)

cc_library(
    name = "desktop_calculators",
    deps = [
        ":template_matching_deps",
        "//mediapipe/calculators/image:opencv_encoded_image_to_image_frame_calculator",
        "//mediapipe/calculators/util:local_file_pattern_contents_calculator",
        "//mediapipe/calculators/video:opencv_video_decoder_calculator",
        "//mediapipe/calculators/video:opencv_video_encoder_calculator",
    ],
)

cc_library(
    name = "mobile_calculators",
    deps = [
        ":template_matching_deps",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
    ],
)

mediapipe_binary_graph(
    name = "mobile_cpu_binary_graph",
    graph = "template_matching_mobile_cpu.pbtxt",
    output_name = "mobile_cpu.binarypb",
    deps = [":mobile_calculators"],
)
