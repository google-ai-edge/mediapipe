#
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
# Description:
#   Working with dense optical flow in mediapipe.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_proto_library(
    name = "optical_flow_field_data_proto",
    srcs = ["optical_flow_field_data.proto"],
)

cc_library(
    name = "optical_flow_field",
    srcs = ["optical_flow_field.cc"],
    hdrs = ["optical_flow_field.h"],
    deps = [
        ":optical_flow_field_data_cc_proto",
        "//mediapipe/framework:type_map",
        "//mediapipe/framework/deps:mathutil",
        "//mediapipe/framework/formats:location",
        "//mediapipe/framework/formats:location_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:point",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

cc_test(
    name = "optical_flow_field_test",
    srcs = ["optical_flow_field_test.cc"],
    linkstatic = 1,
    deps = [
        ":optical_flow_field",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:location_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:gtest_main",
        "//third_party:opencv",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)
