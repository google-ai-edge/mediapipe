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

load("@pybind11_bazel//:build_defs.bzl", "pybind_library")

licenses(["notice"])

package(default_visibility = ["//mediapipe:__subpackages__"])

pybind_library(
    name = "task_runner",
    srcs = ["task_runner.cc"],
    hdrs = ["task_runner.h"],
    deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/gpu:gpu_shared_data_internal",
        "//mediapipe/python/pybind:util",
        "//mediapipe/tasks/cc/core:mediapipe_builtin_op_resolver",
        "//mediapipe/tasks/cc/core:task_runner",
        "@com_google_absl//absl/log:absl_log",
        "@org_tensorflow//tensorflow/lite/core/api:op_resolver",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)
