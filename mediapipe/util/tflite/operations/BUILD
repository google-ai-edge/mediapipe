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

CUSTOM_OPS_PACKAGE = "@org_tensorflow//tensorflow/lite/delegates/gpu/common/mediapipe"

cc_library(
    name = "landmarks_to_transform_matrix",
    srcs = ["landmarks_to_transform_matrix.cc"],
    hdrs = ["landmarks_to_transform_matrix.h"],
    deps = [
        "@org_tensorflow//tensorflow/lite/delegates/gpu/common:types",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
        "@org_tensorflow//tensorflow/lite/kernels:padding",
        "@org_tensorflow//tensorflow/lite/kernels/internal:common",
        "@org_tensorflow//tensorflow/lite/kernels/internal:compatibility",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
        CUSTOM_OPS_PACKAGE + ":landmarks_to_transform_matrix",
    ],
)

cc_library(
    name = "max_pool_argmax",
    srcs = ["max_pool_argmax.cc"],
    hdrs = ["max_pool_argmax.h"],
    deps = [
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
        "@org_tensorflow//tensorflow/lite/kernels:padding",
        "@org_tensorflow//tensorflow/lite/kernels/internal:common",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor_utils",
    ],
)

cc_library(
    name = "max_unpooling",
    srcs = ["max_unpooling.cc"],
    hdrs = ["max_unpooling.h"],
    deps = [
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
        "@org_tensorflow//tensorflow/lite/kernels:padding",
        "@org_tensorflow//tensorflow/lite/kernels/internal:common",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor_utils",
    ],
)

cc_library(
    name = "transform_landmarks",
    srcs = ["transform_landmarks.cc"],
    hdrs = ["transform_landmarks.h"],
    deps = [
        "@org_tensorflow//tensorflow/lite/delegates/gpu/common:types",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
        "@org_tensorflow//tensorflow/lite/kernels:padding",
        "@org_tensorflow//tensorflow/lite/kernels/internal:common",
        "@org_tensorflow//tensorflow/lite/kernels/internal:compatibility",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
        CUSTOM_OPS_PACKAGE + ":transform_landmarks",
    ],
)

cc_library(
    name = "transform_tensor_bilinear",
    srcs = ["transform_tensor_bilinear.cc"],
    hdrs = ["transform_tensor_bilinear.h"],
    deps = [
        "@org_tensorflow//tensorflow/lite/delegates/gpu/common:types",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
        "@org_tensorflow//tensorflow/lite/kernels:padding",
        "@org_tensorflow//tensorflow/lite/kernels/internal:common",
        "@org_tensorflow//tensorflow/lite/kernels/internal:compatibility",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
        CUSTOM_OPS_PACKAGE + ":transform_tensor_bilinear",
    ],
)

cc_library(
    name = "transpose_conv_bias",
    srcs = ["transpose_conv_bias.cc"],
    hdrs = ["transpose_conv_bias.h"],
    deps = [
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
        "@org_tensorflow//tensorflow/lite/kernels:padding",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor_utils",
        "@org_tensorflow//tensorflow/lite/kernels/internal:types",
    ],
)

cc_library(
    name = "resampler",
    srcs = ["resampler.cc"],
    hdrs = ["resampler.h"],
    deps = [
        "@org_tensorflow//tensorflow/lite/c:c_api_types",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)
