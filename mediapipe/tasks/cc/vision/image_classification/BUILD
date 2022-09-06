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
    name = "image_classifier_options_proto",
    srcs = ["image_classifier_options.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/tasks/cc/components:classifier_options_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_proto",
    ],
)

cc_library(
    name = "image_classifier_graph",
    srcs = ["image_classifier_graph.cc"],
    deps = [
        ":image_classifier_options_cc_proto",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:image",
        "//mediapipe/tasks/cc/components:classification_postprocessing",
        "//mediapipe/tasks/cc/components:classification_postprocessing_options_cc_proto",
        "//mediapipe/tasks/cc/components:image_preprocessing",
        "//mediapipe/tasks/cc/components:image_preprocessing_options_cc_proto",
        "//mediapipe/tasks/cc/components/containers:classifications_cc_proto",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/core:model_task_graph",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

cc_library(
    name = "image_classifier",
    srcs = ["image_classifier.cc"],
    hdrs = ["image_classifier.h"],
    deps = [
        ":image_classifier_graph",
        ":image_classifier_options_cc_proto",
        "//mediapipe/framework:packet",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/formats:image",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/containers:classifications_cc_proto",
        "//mediapipe/tasks/cc/core:base_task_api",
        "//mediapipe/tasks/cc/core:task_api_factory",
        "//mediapipe/tasks/cc/core:task_runner",
        "//mediapipe/tasks/cc/core/proto:inference_subgraph_cc_proto",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@org_tensorflow//tensorflow/lite/core/api:op_resolver",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

# TODO: This test fails in OSS
