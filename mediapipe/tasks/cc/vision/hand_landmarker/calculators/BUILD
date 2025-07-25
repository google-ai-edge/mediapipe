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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

mediapipe_proto_library(
    name = "hand_association_calculator_proto",
    srcs = ["hand_association_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "hand_association_calculator",
    srcs = ["hand_association_calculator.cc"],
    deps = [
        ":hand_association_calculator_cc_proto",
        "//mediapipe/calculators/util:association_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:collection_item_id",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:rectangle",
        "//mediapipe/framework/port:status",
        "//mediapipe/util:rectangle_util",
        "@com_google_absl//absl/log:absl_check",
    ],
    alwayslink = 1,
)

cc_library(
    name = "hand_landmarks_deduplication_calculator",
    srcs = ["hand_landmarks_deduplication_calculator.cc"],
    hdrs = ["hand_landmarks_deduplication_calculator.h"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/tasks/cc/components/containers:rect",
        "//mediapipe/tasks/cc/vision/utils:landmarks_duplicates_finder",
        "//mediapipe/tasks/cc/vision/utils:landmarks_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:optional",
    ],
    alwayslink = 1,
)
