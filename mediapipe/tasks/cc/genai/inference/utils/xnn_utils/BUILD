# Copyright 2024 The MediaPipe Authors.
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

# Utilities needed to interacte with XNNPACK.

load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")

DEFAULT_VISIBILITY = [
    "//mediapipe/tasks/cc/genai:__subpackages__",
]

package(default_visibility = DEFAULT_VISIBILITY)

cc_library(
    name = "tensor",
    srcs = ["xnn_tensor.cc"],
    hdrs = ["xnn_tensor.h"],
    deps = [
        ":utils",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc/genai/inference/common:mdspan",
        "@XNNPACK",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

# TODO: move unittest from experimental as well.
cc_library(
    name = "graph_builder",
    srcs = ["graph_builder.cc"],
    hdrs = ["graph_builder.h"],
    deps = [
        ":tensor",
        ":utils",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@XNNPACK",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc/genai/inference/utils/llm_utils:memory_mapped_file",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "benchmark_weight_accessor",
    srcs = ["benchmark_weight_accessor.cc"],
    hdrs = ["benchmark_weight_accessor.h"],
    deps = [
        ":tensor",
        "//mediapipe/framework/port:status",
        "@XNNPACK",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tflite_weight_accessor",
    srcs = ["tflite_weight_accessor.cc"],
    hdrs = ["tflite_weight_accessor.h"],
    deps = [
        ":tensor",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/tasks/cc/genai/inference/utils/llm_utils:memory_mapped_file",
        "@XNNPACK",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

# TODO: move unittest from experimental as well.

cc_library(
    name = "llm_weights",
    srcs = ["llm_weights.cc"],
    hdrs = ["llm_weights.h"],
    deps = [
        ":graph_builder",
        ":pack_weights_cache",
        ":tensor",
        ":tflite_weight_accessor",
        ":utils",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc/genai/inference/proto:llm_params_cc_proto",
        "//mediapipe/tasks/cc/genai/inference/proto:transformer_params_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "llm",
    srcs = ["llm.cc"],
    hdrs = ["llm.h"],
    deps = [
        ":graph_builder",
        ":llm_weights",
        ":sampling",
        ":tensor",
        ":utils",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc/genai/inference/common:mdspan",
        "@XNNPACK",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "llm_builder_factory",
    srcs = ["llm_builder_factory.cc"],
    hdrs = ["llm_builder_factory.h"],
    deps = [
        ":falcon",
        ":graph_builder",
        ":llm",
        ":llm_weights",
        ":phi",
        ":sampling",
        ":stablelm",
        "//mediapipe/tasks/cc/genai/inference/proto:llm_params_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "stablelm",
    srcs = ["stablelm.cc"],
    hdrs = ["stablelm.h"],
    deps = [
        ":llm",
        ":llm_weights",
        ":tensor",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "llm_test",
    size = "medium",
    timeout = "eternal",
    srcs = [
        "llm_test.cc",
    ],
    deps = [
        ":benchmark_weight_accessor",
        ":falcon",
        ":graph_builder",
        ":llm",
        ":llm_weights",
        ":phi",
        ":sampling",
        ":stablelm",
        ":tensor",
        "//mediapipe/framework/port:benchmark",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/tasks/cc/genai/inference/common:mdspan",
        "//mediapipe/tasks/cc/genai/inference/proto:llm_params_cc_proto",
        "//mediapipe/tasks/cc/genai/inference/utils/llm_utils:well_known_models",
        "@XNNPACK",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

flatbuffer_cc_library(
    name = "named_buffer_generated",
    srcs = ["named_buffer.fbs"],
    flatc_args = [
        "--gen-mutable",
    ],
)

cc_library(
    name = "pack_weights_cache",
    srcs = ["pack_weights_cache.cc"],
    hdrs = ["pack_weights_cache.h"],
    deps = [
        ":graph_builder",
        ":named_buffer_generated",
        ":tensor",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc/genai/inference/utils/llm_utils:memory_mapped_file",
        "@XNNPACK",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "falcon",
    srcs = ["falcon.cc"],
    hdrs = ["falcon.h"],
    deps = [
        ":llm",
        ":llm_weights",
        ":tensor",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc/genai/inference/common:mdspan",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "phi",
    srcs = ["phi.cc"],
    hdrs = ["phi.h"],
    deps = [
        ":llm",
        ":llm_weights",
        ":tensor",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "sampling",
    srcs = ["sampling.cc"],
    hdrs = ["sampling.h"],
    deps = [
        ":tensor",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)
