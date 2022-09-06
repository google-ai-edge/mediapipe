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

mediapipe_proto_library(
    name = "segmenter_options_proto",
    srcs = ["segmenter_options.proto"],
)

cc_library(
    name = "classifier_options",
    srcs = ["classifier_options.cc"],
    hdrs = ["classifier_options.h"],
    deps = [":classifier_options_cc_proto"],
)

mediapipe_proto_library(
    name = "classifier_options_proto",
    srcs = ["classifier_options.proto"],
)

mediapipe_proto_library(
    name = "classification_postprocessing_options_proto",
    srcs = ["classification_postprocessing_options.proto"],
    deps = [
        "//mediapipe/calculators/tensor:tensors_to_classification_calculator_proto",
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/tasks/cc/components/calculators:classification_aggregation_calculator_proto",
    ],
)

cc_library(
    name = "classification_postprocessing",
    srcs = ["classification_postprocessing.cc"],
    hdrs = ["classification_postprocessing.h"],
    deps = [
        ":classification_postprocessing_options_cc_proto",
        ":classifier_options_cc_proto",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/core:split_vector_calculator_cc_proto",
        "//mediapipe/calculators/tensor:tensors_dequantization_calculator",
        "//mediapipe/calculators/tensor:tensors_to_classification_calculator",
        "//mediapipe/calculators/tensor:tensors_to_classification_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/calculators:classification_aggregation_calculator",
        "//mediapipe/tasks/cc/components/calculators:classification_aggregation_calculator_cc_proto",
        "//mediapipe/tasks/cc/components/containers:classifications_cc_proto",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/metadata:metadata_extractor",
        "//mediapipe/tasks/metadata:metadata_schema_cc",
        "//mediapipe/util:label_map_cc_proto",
        "//mediapipe/util:label_map_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
    alwayslink = 1,
)
