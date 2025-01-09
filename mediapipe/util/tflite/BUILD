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

load(
    "@org_tensorflow//tensorflow/lite/core/shims:cc_library_with_tflite.bzl",
    "cc_library_with_tflite",
    "cc_test_with_tflite",
)

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
        "//mediapipe/util/tflite/operations:resampler",
        "//mediapipe/util/tflite/operations:transform_landmarks",
        "//mediapipe/util/tflite/operations:transform_tensor_bilinear",
        "//mediapipe/util/tflite/operations:transpose_conv_bias",
        "@com_google_absl//absl/log:absl_check",
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

cc_test(
    name = "error_reporter_test",
    srcs = ["error_reporter_test.cc"],
    deps = [
        ":error_reporter",
        "//mediapipe/framework/port:gtest_main",
    ],
)

# This target has an implementation dependency on TFLite/TFLite-in-GMSCore,
# but it does not have any API dependency on TFLite-in-GMSCore.
cc_library_with_tflite(
    name = "op_resolver",
    srcs = ["op_resolver.cc"],
    hdrs = ["op_resolver.h"],
    tflite_deps = [
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/c:c_api",
        "@org_tensorflow//tensorflow/lite/c:c_api_experimental",  # For c_api_opaque.h
        "@org_tensorflow//tensorflow/lite/c:common",  # For builtin_op_data.h
    ],
    deps = [
        "@org_tensorflow//tensorflow/lite:builtin_ops",
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
                   "//mediapipe/framework/port:logging",
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
        "//mediapipe/gpu:gl_base",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@org_tensorflow//tensorflow/lite/core:framework",
        "@org_tensorflow//tensorflow/lite/core/api",
        "@org_tensorflow//tensorflow/lite/delegates/gpu/common:data_type",
        "@org_tensorflow//tensorflow/lite/delegates/gpu/common:shape",
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
        ":error_reporter",
        "//mediapipe/framework:resources",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/util:resource_util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test_with_tflite(
    name = "tflite_model_loader_test",
    srcs = ["tflite_model_loader_test.cc"],
    data = [
        ":testdata/test_model.tflite",
    ],
    tflite_deps = [
        ":tflite_model_loader",
        "@org_tensorflow//tensorflow/lite:test_util",
    ],
    deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_state",
        "//mediapipe/framework:legacy_calculator_support",
        "//mediapipe/framework:resources",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/tool:tag_map_helper",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
    ],
)

cc_library_with_tflite(
    name = "tflite_signature_reader",
    srcs = ["tflite_signature_reader.cc"],
    hdrs = ["tflite_signature_reader.h"],
    tflite_deps = [
        "@org_tensorflow//tensorflow/lite:framework_stable",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

# TODO b/336756170 - convert test to use cc_test_with_tflite
cc_test(
    name = "tflite_signature_reader_test",
    srcs = ["tflite_signature_reader_test.cc"],
    data = [
        ":testdata/test_single_signature_key_model.tflite",
        ":testdata/test_two_signature_keys_model.tflite",
    ],
    deps = [
        ":tflite_model_loader",
        ":tflite_signature_reader",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:resources",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@org_tensorflow//tensorflow/lite:framework_stable",
        "@org_tensorflow//tensorflow/lite:test_util",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_library_with_tflite(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    tflite_deps = [
        "@org_tensorflow//tensorflow/lite:framework_stable",
        "@org_tensorflow//tensorflow/lite/c:common",
    ],
    visibility = ["//visibility:public"],
)
