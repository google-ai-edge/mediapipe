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

load("@org_tensorflow//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

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

# TODO: Re-evaluate which of these libraries we can avoid making
# cc_library_with_tflite and can be changed back to cc_library.
cc_library_with_tflite(
    name = "cpu_op_resolver",
    srcs = ["cpu_op_resolver.cc"],
    hdrs = ["cpu_op_resolver.h"],
    tflite_deps = [
        "@org_tensorflow//tensorflow/lite:framework_stable",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/port:logging",
        "//mediapipe/util/tflite/operations:landmarks_to_transform_matrix",
        "//mediapipe/util/tflite/operations:max_pool_argmax",
        "//mediapipe/util/tflite/operations:max_unpooling",
        "//mediapipe/util/tflite/operations:transform_landmarks",
        "//mediapipe/util/tflite/operations:transform_tensor_bilinear",
        "//mediapipe/util/tflite/operations:transpose_conv_bias",
        "@org_tensorflow//tensorflow/lite:builtin_op_data",
    ],
    # For using the symbol `MediaPipe_RegisterTfLiteOpResolver` in Python
    # with `tensorflow.lite.python.interpreter.InterpreterWithCustomOps`.
    alwayslink = 1,
)

cc_library(
    name = "error_reporter",
    srcs = ["error_reporter.cc"],
    hdrs = ["error_reporter.h"],
    deps = [
        "@org_tensorflow//tensorflow/lite:minimal_logging",
        "@org_tensorflow//tensorflow/lite:stateful_error_reporter",
        "@org_tensorflow//tensorflow/lite/core/api:error_reporter",
    ],
)

# TODO: Re-evaluate which of these libraries we can avoid making
# cc_library_with_tflite and can be changed back to cc_library.
cc_library_with_tflite(
    name = "op_resolver",
    srcs = ["op_resolver.cc"],
    hdrs = ["op_resolver.h"],
    tflite_deps = [
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
    deps = [
        "@org_tensorflow//tensorflow/lite:builtin_op_data",
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
                   "//mediapipe/framework/port:ret_check",
                   "//mediapipe/framework/port:status",
                   "//mediapipe/framework/port:statusor",
                   "@com_google_absl//absl/strings",
                   "@org_tensorflow//tensorflow/lite:framework",
                   "@org_tensorflow//tensorflow/lite/delegates/gpu:api",
                   "@org_tensorflow//tensorflow/lite/delegates/gpu/common:model",
                   "@org_tensorflow//tensorflow/lite/delegates/gpu/common:model_builder",
                   "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:api2",
               ],
           }) +
           select({
               "//mediapipe:android": [
                   "@org_tensorflow//tensorflow/lite/delegates/gpu/cl:api",
               ],
               "//conditions:default": [],
           }) + [
        "//mediapipe/framework:port",
        "@com_google_absl//absl/status",
        "@org_tensorflow//tensorflow/lite/core/api",
    ],
)

cc_library_with_tflite(
    name = "tflite_model_loader",
    srcs = ["tflite_model_loader.cc"],
    hdrs = ["tflite_model_loader.h"],
    tflite_deps = [
        "@org_tensorflow//tensorflow/lite:framework_stable",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "//mediapipe/util:resource_util",
    ],
)
