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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "ngram_hash_ops_utils",
    srcs = [
        "ngram_hash_ops_utils.cc",
    ],
    hdrs = [
        "ngram_hash_ops_utils.h",
    ],
    deps = [
        "//mediapipe/tasks/cc/text/language_detector/custom_ops/utils/utf",
    ],
)

cc_test(
    name = "ngram_hash_ops_utils_test",
    size = "small",
    srcs = [
        "ngram_hash_ops_utils_test.cc",
    ],
    deps = [
        ":ngram_hash_ops_utils",
        "//mediapipe/framework/port:gtest_main",
    ],
)
