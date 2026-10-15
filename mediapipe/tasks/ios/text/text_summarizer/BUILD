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
    name = "MPPTextSummarizerOptions",
    srcs = ["sources/MPPTextSummarizerOptions.m"],
    hdrs = ["sources/MPPTextSummarizerOptions.h"],
    deps = [
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPTextSummarizerResult",
    srcs = ["sources/MPPTextSummarizerResult.m"],
    hdrs = ["sources/MPPTextSummarizerResult.h"],
    deps = [
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPTextSummarizer",
    srcs = ["sources/MPPTextSummarizer.mm"],
    hdrs = ["sources/MPPTextSummarizer.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
        "-x objective-c++",
    ],
    module_name = "MPPTextSummarizer",
    deps = [
        ":MPPTextSummarizerOptions",
        ":MPPTextSummarizerResult",
        "//mediapipe/tasks/cc/text/text_summarizer",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//third_party/apple_frameworks:Foundation",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)
