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

load("@org_tensorflow//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite", "cc_test_with_tflite")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

cc_library(
    name = "base_options",
    srcs = ["base_options.cc"],
    hdrs = ["base_options.h"],
    deps = [
        "//mediapipe/tasks/cc/core/proto:base_options_cc_proto",
        "//mediapipe/tasks/cc/core/proto:external_file_cc_proto",
        "@com_google_absl//absl/memory",
        "@org_tensorflow//tensorflow/lite/core/api:op_resolver",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "external_file_handler",
    srcs = ["external_file_handler.cc"],
    hdrs = ["external_file_handler.h"],
    deps = [
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/core/proto:external_file_cc_proto",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

# TODO: Switch to use cc_library_with_tflite after the MediaPipe InferenceCalculator
# supports TFLite-in-GMSCore.
cc_library(
    name = "model_task_graph",
    srcs = ["model_task_graph.cc"],
    hdrs = ["model_task_graph.h"],
    deps = [
        ":model_resources",
        ":model_resources_cache",
        ":model_resources_calculator",
        "//mediapipe/calculators/tensor:inference_calculator_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:subgraph",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/port:logging",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/core/proto:acceleration_cc_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_cc_proto",
        "//mediapipe/tasks/cc/core/proto:external_file_cc_proto",
        "//mediapipe/tasks/cc/core/proto:inference_subgraph_cc_proto",
        "//mediapipe/tasks/cc/core/proto:model_resources_calculator_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

# TODO: Enable this test

cc_library_with_tflite(
    name = "model_resources",
    srcs = ["model_resources.cc"],
    hdrs = ["model_resources.h"],
    tflite_deps = [
        "@org_tensorflow//tensorflow/lite/core/shims:builtin_ops",
        "@org_tensorflow//tensorflow/lite/core/shims:framework_stable",
        "@org_tensorflow//tensorflow/lite/core/shims:verifier",
    ],
    deps = [
        ":external_file_handler",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/core/proto:external_file_cc_proto",
        "//mediapipe/tasks/cc/metadata:metadata_extractor",
        "//mediapipe/util:resource_util",
        "//mediapipe/util/tflite:error_reporter",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/lite/core/api:error_reporter",
        "@org_tensorflow//tensorflow/lite/core/api:op_resolver",
    ],
)

cc_test_with_tflite(
    name = "model_resources_test",
    srcs = ["model_resources_test.cc"],
    data = [
        "//mediapipe/tasks/testdata/core:test_models",
    ],
    tflite_deps = [
        ":model_resources",
        "@org_tensorflow//tensorflow/lite/core/shims:cc_shims_test_util",
        "@org_tensorflow//tensorflow/lite/core/shims:builtin_ops",
        "@org_tensorflow//tensorflow/lite/core/shims:framework_stable",
    ],
    deps = [
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/core/proto:external_file_cc_proto",
        "//mediapipe/tasks/cc/metadata:metadata_extractor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@org_tensorflow//tensorflow/lite:mutable_op_resolver",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/core/api:op_resolver",
    ],
)

cc_library_with_tflite(
    name = "model_resources_cache",
    srcs = ["model_resources_cache.cc"],
    hdrs = ["model_resources_cache.h"],
    tflite_deps = [
        ":model_resources",
        "@org_tensorflow//tensorflow/lite/core/shims:builtin_ops",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/tasks/cc:common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/lite/core/api:op_resolver",
    ],
)

cc_library_with_tflite(
    name = "model_resources_calculator",
    srcs = ["model_resources_calculator.cc"],
    tflite_deps = [
        ":model_resources",
        ":model_resources_cache",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:port",
        "//mediapipe/tasks/cc/core/proto:external_file_cc_proto",
        "//mediapipe/tasks/cc/core/proto:model_resources_calculator_cc_proto",
        "//mediapipe/tasks/cc/metadata:metadata_extractor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@org_tensorflow//tensorflow/lite/core/api:op_resolver",
    ],
    alwayslink = 1,
)

cc_test_with_tflite(
    name = "model_resources_calculator_test",
    srcs = ["model_resources_calculator_test.cc"],
    data = [
        "//mediapipe/tasks/testdata/core:test_models",
    ],
    tflite_deps = [
        ":model_resources",
        ":model_resources_cache",
        ":model_resources_calculator",
        "@org_tensorflow//tensorflow/lite/core/shims:cc_shims_test_util",
    ],
    deps = [
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/tasks/cc/core/proto:external_file_cc_proto",
        "//mediapipe/tasks/cc/core/proto:model_resources_calculator_cc_proto",
        "//mediapipe/tasks/cc/metadata:metadata_extractor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/lite/core/api:op_resolver",
    ],
)

cc_library_with_tflite(
    name = "task_runner",
    srcs = ["task_runner.cc"],
    hdrs = ["task_runner.h"],
    tflite_deps = [
        ":model_resources",
        ":model_resources_cache",
        ":model_resources_calculator",
    ],
    deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:name_util",
        "//mediapipe/tasks/cc:common",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@org_tensorflow//tensorflow/lite/core/api:op_resolver",
    ],
)

cc_test_with_tflite(
    name = "task_runner_test",
    srcs = ["task_runner_test.cc"],
    data = [
        "//mediapipe/tasks/testdata/core:test_models",
    ],
    tflite_deps = [
        ":task_runner",
        ":model_resources",
        ":model_resources_cache",
        "@org_tensorflow//tensorflow/lite/core/shims:cc_shims_test_util",
    ],
    deps = [
        "//mediapipe/calculators/core:pass_through_calculator",
        "//mediapipe/calculators/core:side_packet_to_stream_calculator",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:packet",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/tasks/cc/core/proto:external_file_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "base_task_api",
    hdrs = ["base_task_api.h"],
    deps = [
        ":task_runner",
        "//mediapipe/calculators/core:flow_limiter_calculator",
    ],
)

cc_library(
    name = "task_api_factory",
    hdrs = ["task_api_factory.h"],
    deps = [
        ":base_task_api",
        ":model_resources",
        ":task_runner",
        ":utils",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/core/proto:base_options_cc_proto",
        "//mediapipe/tasks/cc/core/proto:external_file_cc_proto",
        "//mediapipe/tasks/cc/core/proto:inference_subgraph_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/lite/core/api:op_resolver",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    visibility = ["//mediapipe/framework:mediapipe_internal"],
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/tasks/metadata:metadata_schema_cc",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
    ],
)
