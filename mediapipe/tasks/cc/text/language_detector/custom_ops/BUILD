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

load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_copts")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

cc_library(
    name = "kmeans_embedding_lookup",
    srcs = ["kmeans_embedding_lookup.cc"],
    hdrs = ["kmeans_embedding_lookup.h"],
    copts = tflite_copts(),
    deps = [
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
    ],
)

cc_test(
    name = "kmeans_embedding_lookup_test",
    size = "small",
    srcs = ["kmeans_embedding_lookup_test.cc"],
    deps = [
        ":kmeans_embedding_lookup",
        "//mediapipe/framework/port:gtest_main",
        "@com_google_absl//absl/log:absl_check",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
    ],
)

cc_library(
    name = "ngram_hash",
    srcs = ["ngram_hash.cc"],
    hdrs = ["ngram_hash.h"],
    copts = tflite_copts(),
    deps = [
        "//mediapipe/tasks/cc/text/language_detector/custom_ops/utils:ngram_hash_ops_utils",
        "//mediapipe/tasks/cc/text/language_detector/custom_ops/utils/hash:murmur",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
    ],
    alwayslink = 1,
)

cc_test(
    name = "ngram_hash_test",
    srcs = ["ngram_hash_test.cc"],
    deps = [
        ":ngram_hash",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/tasks/cc/text/language_detector/custom_ops/utils/hash:murmur",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/types:optional",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
    ],
)
