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

load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/**",
    ]),
)

filegroup(
    name = "config_fbs",
    srcs = ["config.fbs"],
)

flatbuffer_cc_library(
    name = "config",
    srcs = [
        "config.fbs",
    ],
)

flatbuffer_cc_library(
    name = "encoder_config",
    srcs = [
        "encoder_config.fbs",
    ],
    includes = [":config_fbs"],
)

cc_library(
    name = "utils",
    hdrs = [
        "utils.h",
    ],
)

cc_library(
    name = "double_array_trie",
    hdrs = [
        "double_array_trie.h",
    ],
    deps = [
        ":config",
        ":utils",
    ],
)

cc_library(
    name = "double_array_trie_builder",
    srcs = [
        "double_array_trie_builder.cc",
    ],
    hdrs = [
        "double_array_trie_builder.h",
    ],
    deps = ["@darts_clone"],
)

cc_test(
    name = "double_array_trie_test",
    srcs = [
        "double_array_trie_test.cc",
    ],
    deps = [
        ":double_array_trie",
        ":double_array_trie_builder",
        ":encoder_config",
        ":utils",
        "//mediapipe/framework/port:gtest_main",
    ],
)

cc_library(
    name = "sentencepiece_constants",
    hdrs = ["sentencepiece_constants.h"],
)

cc_library(
    name = "model_converter",
    srcs = [
        "model_converter.cc",
    ],
    hdrs = [
        "model_converter.h",
    ],
    deps = [
        ":config",
        ":double_array_trie_builder",
        ":encoder_config",
        ":sentencepiece_constants",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_sentencepiece//:sentencepiece_model_cc_proto",
    ],
)

cc_library(
    name = "optimized_encoder",
    srcs = [
        "optimized_encoder.cc",
    ],
    hdrs = [
        "optimized_encoder.h",
    ],
    deps = [
        ":double_array_trie",
        ":encoder_config",
        ":utils",
    ],
)

cc_library(
    name = "sentencepiece_tokenizer_tflite",
    srcs = ["sentencepiece_tokenizer_tflite.cc"],
    hdrs = ["sentencepiece_tokenizer_tflite.h"],
    deps =
        [
            ":optimized_encoder",
            "@flatbuffers",
            "@org_tensorflow//tensorflow/lite:framework",
            "@org_tensorflow//tensorflow/lite:string_util",
            "@org_tensorflow//tensorflow/lite/c:common",
            "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
            "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
            "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        ],
)

cc_test(
    name = "optimized_encoder_test",
    srcs = [
        "optimized_encoder_test.cc",
    ],
    data = [
        ":testdata",
    ],
    deps = [
        ":double_array_trie_builder",
        ":encoder_config",
        ":model_converter",
        ":optimized_encoder",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:gtest_main",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_sentencepiece//:sentencepiece_cc_proto",
        "@com_google_sentencepiece//:sentencepiece_processor",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)
