# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

mediapipe_proto_library(
    name = "image_segmenter_options_proto",
    srcs = ["image_segmenter_options.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/tasks/cc/components:segmenter_options_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_proto",
    ],
)

cc_library(
    name = "image_segmenter",
    srcs = ["image_segmenter.cc"],
    hdrs = ["image_segmenter.h"],
    deps = [
        ":image_segmenter_graph",
        ":image_segmenter_options_cc_proto",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/formats:image",
        "//mediapipe/tasks/cc/core:base_task_api",
        "//mediapipe/tasks/cc/core:task_api_factory",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@org_tensorflow//tensorflow/lite/core/api:op_resolver",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "image_segmenter_graph",
    srcs = ["image_segmenter_graph.cc"],
    deps = [
        ":image_segmenter_options_cc_proto",
        "//mediapipe/calculators/core:merge_to_vector_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components:image_preprocessing",
        "//mediapipe/tasks/cc/components:image_preprocessing_options_cc_proto",
        "//mediapipe/tasks/cc/components:segmenter_options_cc_proto",
        "//mediapipe/tasks/cc/components/calculators/tensor:tensors_to_segmentation_calculator",
        "//mediapipe/tasks/cc/components/calculators/tensor:tensors_to_segmentation_calculator_cc_proto",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/core:model_task_graph",
        "//mediapipe/tasks/cc/core/proto:inference_subgraph_cc_proto",
        "//mediapipe/tasks/cc/metadata:metadata_extractor",
        "//mediapipe/tasks/metadata:metadata_schema_cc",
        "//mediapipe/util:label_map_cc_proto",
        "//mediapipe/util:label_map_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
    alwayslink = 1,
)

cc_library(
    name = "custom_op_resolvers",
    srcs = ["custom_op_resolvers.cc"],
    hdrs = ["custom_op_resolvers.h"],
    deps = [
        "//mediapipe/util/tflite/operations:landmarks_to_transform_matrix",
        "//mediapipe/util/tflite/operations:max_pool_argmax",
        "//mediapipe/util/tflite/operations:max_unpooling",
        "//mediapipe/util/tflite/operations:transform_landmarks",
        "//mediapipe/util/tflite/operations:transform_tensor_bilinear",
        "//mediapipe/util/tflite/operations:transpose_conv_bias",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

# TODO: This test fails in OSS
