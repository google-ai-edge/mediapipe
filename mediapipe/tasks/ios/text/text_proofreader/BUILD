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

load("@rules_cc//cc:objc_library.bzl", "objc_library")

package(default_visibility = [
    "//mediapipe/tasks:internal",
])

licenses(["notice"])

exports_files(glob(["sources/*.h"]))

objc_library(
    name = "MPPTextProofreaderOptions",
    srcs = ["sources/MPPTextProofreaderOptions.m"],
    hdrs = ["sources/MPPTextProofreaderOptions.h"],
    deps = [
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPTextProofreaderResult",
    srcs = ["sources/MPPTextProofreaderResult.m"],
    hdrs = ["sources/MPPTextProofreaderResult.h"],
    deps = [
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPTextProofreader",
    srcs = ["sources/MPPTextProofreader.mm"],
    hdrs = ["sources/MPPTextProofreader.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
        "-x objective-c++",
    ],
    module_name = "MPPTextProofreader",
    deps = [
        ":MPPTextProofreaderOptions",
        ":MPPTextProofreaderResult",
        "//mediapipe/tasks/cc/text/text_proofreader",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//third_party/apple_frameworks:Foundation",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)
