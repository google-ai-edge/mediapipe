# Copyright 2026 The MediaPipe Authors.
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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "yolo_object_detector_graph_utils",
    srcs = ["yolo_object_detector_graph_utils.cc"],
    hdrs = ["yolo_object_detector_graph_utils.h"],
    deps = [
        "//mediapipe/framework/port:file_helpers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@org_tensorflow//tensorflow/lite:model_builder",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "yolo_object_detector_graph_utils_test",
    srcs = ["yolo_object_detector_graph_utils_test.cc"],
    linkstatic = 1,
    deps = [
        ":yolo_object_detector_graph_utils",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:status_matchers",
    ],
)
