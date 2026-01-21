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

load("@rules_cc//cc:objc_library.bzl", "objc_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

objc_library(
    name = "MPPCommonUtils",
    srcs = ["sources/MPPCommonUtils.mm"],
    hdrs = ["sources/MPPCommonUtils.h"],
    deps = [
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//third_party/apple_frameworks:Foundation",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
    ],
)

objc_library(
    name = "NSStringHelpers",
    srcs = ["sources/NSString+Helpers.mm"],
    hdrs = ["sources/NSString+Helpers.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
    ],
    deps = ["//third_party/apple_frameworks:Foundation"],
)
