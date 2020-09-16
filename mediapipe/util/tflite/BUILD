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

licenses(["notice"])

package(default_visibility = [
    "//mediapipe:__subpackages__",
])

cc_library(
    name = "config",
    hdrs = ["config.h"],
    features = ["-parse_headers"],
    deps = [
        "//mediapipe/framework:calculator_framework",
    ],
)

cc_library(
    name = "cpu_op_resolver",
    srcs = ["cpu_op_resolver.cc"],
    hdrs = ["cpu_op_resolver.h"],
    deps = [
        "//mediapipe/util/tflite/operations:max_pool_argmax",
        "//mediapipe/util/tflite/operations:max_unpooling",
        "//mediapipe/util/tflite/operations:transpose_conv_bias",
        "@org_tensorflow//tensorflow/lite:builtin_op_data",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "op_resolver",
    srcs = ["op_resolver.cc"],
    hdrs = ["op_resolver.h"],
    deps = [
        "@org_tensorflow//tensorflow/lite:builtin_op_data",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "tflite_gpu_runner",
    srcs = select({
        "//mediapipe:ios": [],
        "//mediapipe:macos": [],
        "//conditions:default": ["tflite_gpu_runner.cc"],
    }),
    hdrs = select({
        "//mediapipe:ios": [],
        "//mediapipe:macos": [],
        "//conditions:default": ["tflite_gpu_runner.h"],
    }),
    deps = select({
        "//mediapipe:ios": [],
        "//mediapipe:macos": [],
        "//conditions:default": [
            "@com_google_absl//absl/strings",
            "//mediapipe/framework/port:ret_check",
            "//mediapipe/framework/port:status",
            "//mediapipe/framework/port:statusor",
            "@org_tensorflow//tensorflow/lite:framework",
            "@org_tensorflow//tensorflow/lite/delegates/gpu:api",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/common:model",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/common/testing:tflite_model_reader",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:api2",
        ],
        "//mediapipe:android": [
            "@com_google_absl//absl/strings",
            "//mediapipe/framework/port:ret_check",
            "//mediapipe/framework/port:status",
            "//mediapipe/framework/port:statusor",
            "@org_tensorflow//tensorflow/lite:framework",
            "@org_tensorflow//tensorflow/lite/delegates/gpu:api",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/cl:api",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/common:model",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/common/testing:tflite_model_reader",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:api2",
        ],
    }) + ["@org_tensorflow//tensorflow/lite/core/api"],
)
