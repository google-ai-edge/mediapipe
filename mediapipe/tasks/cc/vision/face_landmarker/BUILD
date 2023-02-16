# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
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

package(default_visibility = [
    "//mediapipe/tasks:internal",
])

licenses(["notice"])

cc_library(
    name = "face_blendshapes_graph",
    srcs = ["face_blendshapes_graph.cc"],
    deps = [
        "//mediapipe/calculators/core:split_proto_list_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/core:split_vector_calculator_cc_proto",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/tensor:landmarks_to_tensor_calculator",
        "//mediapipe/calculators/tensor:landmarks_to_tensor_calculator_cc_proto",
        "//mediapipe/calculators/tensor:tensors_to_classification_calculator",
        "//mediapipe/calculators/tensor:tensors_to_classification_calculator_cc_proto",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc/core:model_asset_bundle_resources",
        "//mediapipe/tasks/cc/core:model_resources_cache",
        "//mediapipe/tasks/cc/core:model_task_graph",
        "//mediapipe/tasks/cc/vision/face_landmarker/proto:face_blendshapes_graph_options_cc_proto",
    ],
    alwayslink = 1,
)

cc_library(
    name = "face_landmarks_detector_graph",
    srcs = ["face_landmarks_detector_graph.cc"],
    deps = [
        "//mediapipe/calculators/core:begin_loop_calculator",
        "//mediapipe/calculators/core:end_loop_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/core:split_vector_calculator_cc_proto",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/tensor:tensors_to_floats_calculator",
        "//mediapipe/calculators/tensor:tensors_to_floats_calculator_cc_proto",
        "//mediapipe/calculators/tensor:tensors_to_landmarks_calculator",
        "//mediapipe/calculators/tensor:tensors_to_landmarks_calculator_cc_proto",
        "//mediapipe/calculators/util:detections_to_rects_calculator",
        "//mediapipe/calculators/util:detections_to_rects_calculator_cc_proto",
        "//mediapipe/calculators/util:landmark_letterbox_removal_calculator",
        "//mediapipe/calculators/util:landmark_projection_calculator",
        "//mediapipe/calculators/util:landmarks_to_detection_calculator",
        "//mediapipe/calculators/util:rect_transformation_calculator",
        "//mediapipe/calculators/util:rect_transformation_calculator_cc_proto",
        "//mediapipe/calculators/util:thresholding_calculator",
        "//mediapipe/calculators/util:thresholding_calculator_cc_proto",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/processors:image_preprocessing_graph",
        "//mediapipe/tasks/cc/components/utils:gate",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/core:model_task_graph",
        "//mediapipe/tasks/cc/core:utils",
        "//mediapipe/tasks/cc/vision/face_landmarker/proto:face_landmarks_detector_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/utils:image_tensor_specs",
    ],
    alwayslink = 1,
)
