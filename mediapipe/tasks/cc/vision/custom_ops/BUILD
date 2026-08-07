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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "fused_batch_norm",
    srcs = ["fused_batch_norm.cc"],
    hdrs = ["fused_batch_norm.h"],
    deps =
        [
            "@eigen//:eigen3",
            "@litert//tflite:framework",
            "@litert//tflite/c:common",
            "@litert//tflite/core/c:private_common",
            "@litert//tflite/kernels:kernel_util",
            "@litert//tflite/kernels/internal:tensor",
        ],
)
