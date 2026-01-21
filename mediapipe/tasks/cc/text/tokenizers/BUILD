# Copyright 2022 The MediaPipe Authors.
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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "tokenizer",
    hdrs = [
        "tokenizer.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "bert_tokenizer",
    srcs = [
        "bert_tokenizer.cc",
    ],
    hdrs = [
        "bert_tokenizer.h",
    ],
    deps = [
        ":tokenizer",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/tasks/cc/text/utils:vocab_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_googlesource_code_re2//:re2",
        "@org_tensorflow_text//tensorflow_text/core/kernels:regex_split",
        "@org_tensorflow_text//tensorflow_text/core/kernels:wordpiece_tokenizer",
    ],
)

cc_test(
    name = "bert_tokenizer_test",
    srcs = ["bert_tokenizer_test.cc"],
    data = [
        "//mediapipe/tasks/testdata/text:vocab_files",
    ],
    linkopts = ["-ldl"],
    deps = [
        ":bert_tokenizer",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/tasks/cc/core:utils",
    ],
)

cc_library(
    name = "sentencepiece_tokenizer",
    hdrs = [
        "sentencepiece_tokenizer.h",
    ],
    deps = [
        ":tokenizer",
        "//mediapipe/framework/port:logging",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@com_google_sentencepiece//:sentencepiece_processor",
    ],
)

cc_test(
    name = "sentencepiece_tokenizer_test",
    srcs = ["sentencepiece_tokenizer_test.cc"],
    data = [
        "//mediapipe/tasks/testdata/text:albert_model",
    ],
    deps = [
        ":sentencepiece_tokenizer",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/tasks/cc/core:utils",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_sentencepiece//:sentencepiece_processor",
    ],
)

cc_library(
    name = "tokenizer_utils",
    srcs = ["tokenizer_utils.cc"],
    hdrs = [
        "tokenizer_utils.h",
    ],
    deps = [
        ":bert_tokenizer",
        ":regex_tokenizer",
        ":sentencepiece_tokenizer",
        ":tokenizer",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/metadata:metadata_extractor",
        "//mediapipe/tasks/metadata:metadata_schema_cc",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "tokenizer_utils_test",
    srcs = ["tokenizer_utils_test.cc"],
    data = [
        "//mediapipe/tasks/testdata/text:albert_model",
        "//mediapipe/tasks/testdata/text:mobile_bert_model",
        "//mediapipe/tasks/testdata/text:text_classifier_models",
        "@com_google_absl//absl/log:absl_check",
    ],
    linkopts = ["-ldl"],
    deps = [
        ":bert_tokenizer",
        ":regex_tokenizer",
        ":sentencepiece_tokenizer",
        ":tokenizer_utils",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/core:utils",
        "//mediapipe/tasks/cc/metadata:metadata_extractor",
        "//mediapipe/tasks/metadata:metadata_schema_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_sentencepiece//:sentencepiece_processor",
    ],
)

cc_library(
    name = "regex_tokenizer",
    srcs = [
        "regex_tokenizer.cc",
    ],
    hdrs = [
        "regex_tokenizer.h",
    ],
    deps = [
        ":tokenizer",
        "//mediapipe/tasks/cc/text/utils:vocab_utils",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/strings",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_test(
    name = "regex_tokenizer_test",
    srcs = ["regex_tokenizer_test.cc"],
    data = [
        "//mediapipe/tasks/testdata/text:regex_tokenizer_files",
    ],
    deps = [
        ":regex_tokenizer",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/tasks/cc/core:utils",
    ],
)
