# Copyright 2023 The MediaPipe Authors.
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

exports_files(glob(["sources/*.h"]))

objc_library(
    name = "MPPInteractiveSegmenterLegacyOptions",
    srcs = ["sources/MPPInteractiveSegmenterLegacyOptions.m"],
    hdrs = ["sources/MPPInteractiveSegmenterLegacyOptions.h"],
    deps = [
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPInteractiveSegmenterLegacyResult",
    srcs = ["sources/MPPInteractiveSegmenterLegacyResult.m"],
    hdrs = ["sources/MPPInteractiveSegmenterLegacyResult.h"],
    deps = [
        "//mediapipe/tasks/ios/core:MPPTaskResult",
        "//mediapipe/tasks/ios/vision/core:MPPMask",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPInteractiveSegmenterLegacy",
    srcs = ["sources/MPPInteractiveSegmenterLegacy.mm"],
    hdrs = ["sources/MPPInteractiveSegmenterLegacy.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
        "-x objective-c++",
    ],
    module_name = "MPPInteractiveSegmenterLegacy",
    deps = [
        ":MPPInteractiveSegmenterLegacyOptions",
        ":MPPInteractiveSegmenterLegacyResult",
        "//mediapipe/tasks/c/core:common",
        "//mediapipe/tasks/c/core:mp_status",
        "//mediapipe/tasks/c/vision/core:image_processing_options",
        "//mediapipe/tasks/c/vision/interactive_segmenter_legacy:interactive_segmenter_legacy_lib",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/components/containers:MPPRegionOfInterest",
        "//mediapipe/tasks/ios/components/containers/utils:MPPRegionOfInterestHelpers",
        "//mediapipe/tasks/ios/vision/core:MPPImage",
        "//mediapipe/tasks/ios/vision/core:MPPVisionTaskRunner",
        "//mediapipe/tasks/ios/vision/interactive_segmenter_legacy/utils:MPPInteractiveSegmenterLegacyOptionsHelpers",
        "//mediapipe/tasks/ios/vision/interactive_segmenter_legacy/utils:MPPInteractiveSegmenterLegacyResultHelpers",
        "//third_party/apple_frameworks:Foundation",
    ],
)
