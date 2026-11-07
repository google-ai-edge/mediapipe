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

objc_library(
    name = "MPPImageSegmenterOptionsHelpers",
    srcs = ["sources/MPPImageSegmenterOptions+Helpers.mm"],
    hdrs = ["sources/MPPImageSegmenterOptions+Helpers.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
        "-x objective-c++",
    ],
    deps = [
        "//mediapipe/tasks/c/vision/image_segmenter:image_segmenter_lib",
        "//mediapipe/tasks/ios/core/utils:MPPBaseOptionsHelpers",
        "//mediapipe/tasks/ios/vision/image_segmenter:MPPImageSegmenterOptions",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPImageSegmenterResultHelpers",
    srcs = ["sources/MPPImageSegmenterResult+Helpers.mm"],
    hdrs = ["sources/MPPImageSegmenterResult+Helpers.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
        "-x objective-c++",
    ],
    deps = [
        "//mediapipe/tasks/c/vision/core:image",
        "//mediapipe/tasks/c/vision/image_segmenter:image_segmenter_result",
        "//mediapipe/tasks/ios/vision/image_segmenter:MPPImageSegmenterResult",
        "//third_party/apple_frameworks:Foundation",
    ],
)
