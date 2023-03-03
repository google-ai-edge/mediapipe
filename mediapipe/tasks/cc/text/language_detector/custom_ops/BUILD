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
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
    ],
)
