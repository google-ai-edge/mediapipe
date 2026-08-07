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
#
# RaggedTensors support in TFLite

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "ragged_tensor_to_tensor_tflite",
    srcs = ["ragged_tensor_to_tensor_tflite.cc"],
    hdrs = ["ragged_tensor_to_tensor_tflite.h"],
    deps = [
        "@flatbuffers",
        "@litert//tflite:framework",
        "@litert//tflite/c:common",
        "@litert//tflite/kernels:builtin_ops",
        "@litert//tflite/kernels:kernel_util",
        "@litert//tflite/kernels/internal:tensor",
        "@litert//tflite/kernels/internal:types",
        "@org_tensorflow//tensorflow/core/util:ragged_to_dense_util_common",
    ],
)

cc_test(
    name = "ragged_tensor_to_tensor_tflite_test",
    srcs = ["ragged_tensor_to_tensor_tflite_test.cc"],
    deps = [
        ":ragged_tensor_to_tensor_tflite",
        "//mediapipe/framework/port:gtest_main",
        "@flatbuffers",
        "@litert//tflite:framework",
        "@litert//tflite/c:common",
        "@litert//tflite/kernels:test_util",
        "@litert//tflite/kernels/internal:tensor",
        "@litert//tflite/schema:schema_fbs",
    ],
)
