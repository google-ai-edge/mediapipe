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

load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

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
