# Copyright 2019 The MediaPipe Authors.
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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

licenses(["notice"])

package(default_visibility = [
    "//research/aimatter/tflite:__subpackages__",
    "//mediapipe:__subpackages__",
    # For automated benchmarking of Camera models by TFLite team.
    "//learning/brain/models/app_benchmarks/camera_models:__subpackages__",
])

CUSTOM_OPS_PACKAGE = "@litert//tflite/delegates/gpu/common/mediapipe"

cc_library(
    name = "landmarks_to_transform_matrix",
    srcs = ["landmarks_to_transform_matrix.cc"],
    hdrs = ["landmarks_to_transform_matrix.h"],
    deps = [
        "@litert//tflite/delegates/gpu/common:types",
        "@litert//tflite/kernels:kernel_util",
        "@litert//tflite/kernels:padding",
        "@litert//tflite/kernels/internal:common",
        "@litert//tflite/kernels/internal:compatibility",
        "@litert//tflite/kernels/internal:tensor",
        "@litert//tflite/schema:schema_fbs",
        CUSTOM_OPS_PACKAGE + ":landmarks_to_transform_matrix",
    ],
)

cc_library(
    name = "max_pool_argmax",
    srcs = ["max_pool_argmax.cc"],
    hdrs = ["max_pool_argmax.h"],
    deps = [
        "@litert//tflite/kernels:kernel_util",
        "@litert//tflite/kernels:padding",
        "@litert//tflite/kernels/internal:common",
        "@litert//tflite/kernels/internal:tensor",
        "@litert//tflite/kernels/internal:tensor_utils",
    ],
)

cc_library(
    name = "max_unpooling",
    srcs = ["max_unpooling.cc"],
    hdrs = ["max_unpooling.h"],
    deps = [
        "@litert//tflite/kernels:kernel_util",
        "@litert//tflite/kernels:padding",
        "@litert//tflite/kernels/internal:common",
        "@litert//tflite/kernels/internal:tensor",
        "@litert//tflite/kernels/internal:tensor_utils",
    ],
)

cc_library(
    name = "transform_landmarks",
    srcs = ["transform_landmarks.cc"],
    hdrs = ["transform_landmarks.h"],
    deps = [
        "@litert//tflite/delegates/gpu/common:types",
        "@litert//tflite/kernels:kernel_util",
        "@litert//tflite/kernels:padding",
        "@litert//tflite/kernels/internal:common",
        "@litert//tflite/kernels/internal:compatibility",
        "@litert//tflite/kernels/internal:tensor",
        "@litert//tflite/schema:schema_fbs",
        CUSTOM_OPS_PACKAGE + ":transform_landmarks",
    ],
)

cc_library(
    name = "transform_tensor_bilinear",
    srcs = ["transform_tensor_bilinear.cc"],
    hdrs = ["transform_tensor_bilinear.h"],
    deps = [
        "@litert//tflite/delegates/gpu/common:types",
        "@litert//tflite/kernels:kernel_util",
        "@litert//tflite/kernels:padding",
        "@litert//tflite/kernels/internal:common",
        "@litert//tflite/kernels/internal:compatibility",
        "@litert//tflite/kernels/internal:tensor",
        "@litert//tflite/schema:schema_fbs",
        CUSTOM_OPS_PACKAGE + ":transform_tensor_bilinear",
    ],
)

cc_library(
    name = "transpose_conv_bias",
    srcs = ["transpose_conv_bias.cc"],
    hdrs = ["transpose_conv_bias.h"],
    deps = [
        "@litert//tflite/kernels:kernel_util",
        "@litert//tflite/kernels:padding",
        "@litert//tflite/kernels/internal:tensor",
        "@litert//tflite/kernels/internal:tensor_utils",
        "@litert//tflite/kernels/internal:types",
    ],
)

cc_library(
    name = "resampler",
    srcs = ["resampler.cc"],
    hdrs = ["resampler.h"],
    deps = [
        "@litert//tflite/c:c_api_types",
        "@litert//tflite/c:common",
        "@litert//tflite/kernels:kernel_util",
        "@litert//tflite/kernels/internal:tensor",
        "@litert//tflite/schema:schema_fbs",
    ],
)
