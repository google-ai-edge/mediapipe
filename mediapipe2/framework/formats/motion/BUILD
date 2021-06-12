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

licenses(["notice"])

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library")

package(default_visibility = ["//visibility:private"])

proto_library(
    name = "optical_flow_field_data_proto",
    srcs = ["optical_flow_field_data.proto"],
    visibility = ["//visibility:public"],
)

mediapipe_cc_proto_library(
    name = "optical_flow_field_data_cc_proto",
    srcs = ["optical_flow_field_data.proto"],
    visibility = ["//visibility:public"],
    deps = [":optical_flow_field_data_proto"],
)

cc_library(
    name = "optical_flow_field",
    srcs = ["optical_flow_field.cc"],
    hdrs = ["optical_flow_field.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/framework:type_map",
        "//mediapipe/framework/deps:mathutil",
        "//mediapipe/framework/formats:location",
        "//mediapipe/framework/formats/motion:optical_flow_field_data_cc_proto",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:point",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
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
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "@com_google_absl//absl/flags:flag",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)
