# Copyright 2024 The MediaPipe Authors.
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

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "litert_service",
    hdrs = ["litert_service.h"],
    deps = [
        "//mediapipe/framework:graph_service",
    ],
)

cc_library(
    name = "litert_utils",
    srcs = ["litert_utils.cc"],
    hdrs = ["litert_utils.h"],
    deps = [
        "//mediapipe/framework:memory_manager",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_macros",
        "@litert//litert/cc:litert_model",
        "@litert//litert/cc:litert_tensor_buffer",
    ],
)

cc_test(
    name = "litert_utils_test",
    srcs = ["litert_utils_test.cc"],
    deps = [
        ":litert_utils",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:gtest_main",
        "@com_google_absl//absl/status",
        "@litert//litert/c:litert_environment",
        "@litert//litert/c:litert_tensor_buffer",
        "@litert//litert/cc:litert_element_type",
        "@litert//litert/cc:litert_layout",
        "@litert//litert/cc:litert_model",
    ],
)

cc_test(
    name = "litert_dynamic_resize_test",
    srcs = ["litert_dynamic_resize_test.cc"],
    deps = [
        ":litert_utils",
        "@com_google_googletest//:gtest_main",
    ],
)
