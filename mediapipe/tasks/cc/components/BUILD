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
    name = "image_preprocessing_options_proto",
    srcs = ["image_preprocessing_options.proto"],
    deps = [
        "//mediapipe/calculators/tensor:image_to_tensor_calculator_proto",
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "image_preprocessing",
    srcs = ["image_preprocessing.cc"],
    hdrs = ["image_preprocessing.h"],
    deps = [
        ":image_preprocessing_options_cc_proto",
        "//mediapipe/calculators/core:pass_through_calculator",
        "//mediapipe/calculators/image:image_clone_calculator",
        "//mediapipe/calculators/image:image_clone_calculator_cc_proto",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/vision/utils:image_tensor_specs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
    alwayslink = 1,
)

# TODO: Enable this test

cc_library(
    name = "embedder_options",
    srcs = ["embedder_options.cc"],
    hdrs = ["embedder_options.h"],
    deps = ["//mediapipe/tasks/cc/components/proto:embedder_options_cc_proto"],
)

cc_library(
    name = "embedding_postprocessing_graph",
    srcs = ["embedding_postprocessing_graph.cc"],
    hdrs = ["embedding_postprocessing_graph.h"],
    deps = [
        "//mediapipe/calculators/tensor:tensors_dequantization_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/tool:options_map",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/calculators:tensors_to_embeddings_calculator",
        "//mediapipe/tasks/cc/components/calculators:tensors_to_embeddings_calculator_cc_proto",
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_cc_proto",
        "//mediapipe/tasks/cc/components/proto:embedder_options_cc_proto",
        "//mediapipe/tasks/cc/components/proto:embedding_postprocessing_graph_options_cc_proto",
        "//mediapipe/tasks/cc/components/utils:source_or_node_output",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/metadata:metadata_extractor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
    alwayslink = 1,
)
