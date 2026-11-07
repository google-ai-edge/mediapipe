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
    name = "MPPImageSegmenterResult",
    srcs = ["sources/MPPImageSegmenterResult.m"],
    hdrs = ["sources/MPPImageSegmenterResult.h"],
    deps = [
        "//mediapipe/tasks/ios/core:MPPTaskResult",
        "//mediapipe/tasks/ios/vision/core:MPPMask",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPImageSegmenterOptions",
    srcs = ["sources/MPPImageSegmenterOptions.m"],
    hdrs = ["sources/MPPImageSegmenterOptions.h"],
    deps = [
        ":MPPImageSegmenterResult",
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//mediapipe/tasks/ios/vision/core:MPPRunningMode",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPImageSegmenter",
    srcs = ["sources/MPPImageSegmenter.mm"],
    hdrs = ["sources/MPPImageSegmenter.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
        "-x objective-c++",
    ],
    module_name = "MPPImageSegmenter",
    deps = [
        ":MPPImageSegmenterOptions",
        ":MPPImageSegmenterResult",
        "//mediapipe/tasks/c/core:common",
        "//mediapipe/tasks/c/vision/core:image_processing_options",
        "//mediapipe/tasks/c/vision/image_segmenter:image_segmenter_lib",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/vision/core:MPPImage",
        "//mediapipe/tasks/ios/vision/core:MPPVisionTaskRunner",
        "//mediapipe/tasks/ios/vision/image_segmenter/utils:MPPImageSegmenterOptionsHelpers",
        "//mediapipe/tasks/ios/vision/image_segmenter/utils:MPPImageSegmenterResultHelpers",
        "//third_party/apple_frameworks:Foundation",
    ],
)
