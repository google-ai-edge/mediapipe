# Copyright 2023 The MediaPipe Authors.
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

licenses(["notice"])

package(default_visibility = ["//mediapipe/tasks:internal"])

cc_library(
    name = "conditioned_image_graph",
    srcs = ["conditioned_image_graph.cc"],
    deps = [
        "//mediapipe/calculators/core:get_vector_item_calculator",
        "//mediapipe/calculators/core:get_vector_item_calculator_cc_proto",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:flat_color_image_calculator",
        "//mediapipe/calculators/util:flat_color_image_calculator_cc_proto",
        "//mediapipe/calculators/util:landmarks_to_render_data_calculator",
        "//mediapipe/calculators/util:landmarks_to_render_data_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/tasks/cc/core:model_task_graph",
        "//mediapipe/tasks/cc/vision/face_detector/proto:face_detector_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker_graph",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarks_connections",
        "//mediapipe/tasks/cc/vision/face_landmarker/proto:face_landmarker_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/image_generator/proto:conditioned_image_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/image_segmenter:image_segmenter_graph",
        "//mediapipe/tasks/cc/vision/image_segmenter/proto:image_segmenter_graph_options_cc_proto",
        "//mediapipe/util:color_cc_proto",
        "//mediapipe/util:image_frame_util",
        "//mediapipe/util:render_data_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "image_generator_graph",
    srcs = ["image_generator_graph.cc"],
    deps = [
        ":conditioned_image_graph",
        "//mediapipe/calculators/core:pass_through_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator_cc_proto",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator_cc_proto",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/tensor:inference_calculator_cc_proto",
        "//mediapipe/calculators/util:from_image_calculator",
        "//mediapipe/calculators/util:to_image_calculator",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:stream_handler_cc_proto",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:switch_container",
        "//mediapipe/framework/tool:switch_container_cc_proto",
        "//mediapipe/tasks/cc/core:model_asset_bundle_resources",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/core:model_task_graph",
        "//mediapipe/tasks/cc/core/proto:external_file_cc_proto",
        "//mediapipe/tasks/cc/vision/image_generator/diffuser:diffusion_plugins_output_calculator",
        "//mediapipe/tasks/cc/vision/image_generator/diffuser:stable_diffusion_iterate_calculator",
        "//mediapipe/tasks/cc/vision/image_generator/diffuser:stable_diffusion_iterate_calculator_cc_proto",
        "//mediapipe/tasks/cc/vision/image_generator/proto:conditioned_image_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/image_generator/proto:control_plugin_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/image_generator/proto:image_generator_graph_options_cc_proto",
        "//mediapipe/util:graph_builder_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "image_generator_result",
    hdrs = ["image_generator_result.h"],
    deps = ["//mediapipe/framework/formats:image"],
)

cc_library(
    name = "image_generator",
    srcs = ["image_generator.cc"],
    hdrs = ["image_generator.h"],
    deps = [
        ":image_generator_graph",
        ":image_generator_result",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/tasks/cc/core:base_options",
        "//mediapipe/tasks/cc/core:task_runner",
        "//mediapipe/tasks/cc/core/proto:external_file_cc_proto",
        "//mediapipe/tasks/cc/vision/core:base_vision_task_api",
        "//mediapipe/tasks/cc/vision/core:vision_task_api_factory",
        "//mediapipe/tasks/cc/vision/face_detector/proto:face_detector_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/face_landmarker",
        "//mediapipe/tasks/cc/vision/face_landmarker/proto:face_landmarker_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/face_landmarker/proto:face_landmarks_detector_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/image_generator/diffuser:stable_diffusion_iterate_calculator_cc_proto",
        "//mediapipe/tasks/cc/vision/image_generator/proto:conditioned_image_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/image_generator/proto:control_plugin_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/image_generator/proto:image_generator_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/image_segmenter",
        "//mediapipe/tasks/cc/vision/image_segmenter/proto:image_segmenter_graph_options_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)
