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
        "//mediapipe/calculators/tensor:inference_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/gpu:gpu_origin_cc_proto",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/core/proto:acceleration_cc_proto",
        "//mediapipe/tasks/cc/vision/utils:image_tensor_specs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
    alwayslink = 1,
)

# TODO: Enable this test

# TODO: Investigate rewriting the build rule to only link
# the Bert Preprocessor if it's needed.
cc_library(
    name = "text_preprocessing_graph",
    srcs = ["text_preprocessing_graph.cc"],
    hdrs = ["text_preprocessing_graph.h"],
    deps = [
        "//mediapipe/calculators/tensor:bert_preprocessor_calculator",
        "//mediapipe/calculators/tensor:bert_preprocessor_calculator_cc_proto",
        "//mediapipe/calculators/tensor:regex_preprocessor_calculator",
        "//mediapipe/calculators/tensor:regex_preprocessor_calculator_cc_proto",
        "//mediapipe/calculators/tensor:text_to_tensor_calculator",
        "//mediapipe/framework:subgraph",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/tasks/cc/components/proto:text_preprocessing_graph_options_cc_proto",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/metadata:metadata_extractor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)
