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

licenses(["notice"])  # Apache 2.0

package(default_visibility = [
    "//mediapipe:__subpackages__",
])

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
