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
    name = "MPPInteractiveSegmenterOptions",
    srcs = ["sources/MPPInteractiveSegmenterOptions.m"],
    hdrs = ["sources/MPPInteractiveSegmenterOptions.h"],
    deps = [
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPInteractiveSegmenterResult",
    srcs = ["sources/MPPInteractiveSegmenterResult.m"],
    hdrs = ["sources/MPPInteractiveSegmenterResult.h"],
    deps = [
        "//mediapipe/tasks/ios/core:MPPTaskResult",
        "//mediapipe/tasks/ios/vision/core:MPPMask",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPInteractiveSegmenter",
    srcs = ["sources/MPPInteractiveSegmenter.mm"],
    hdrs = ["sources/MPPInteractiveSegmenter.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
        "-x objective-c++",
    ],
    module_name = "MPPInteractiveSegmenter",
    deps = [
        ":MPPInteractiveSegmenterOptions",
        ":MPPInteractiveSegmenterResult",
        "//mediapipe/tasks/cc/vision/image_segmenter/calculators:tensors_to_segmentation_calculator",
        "//mediapipe/tasks/cc/vision/image_segmenter/calculators:tensors_to_segmentation_calculator_cc_proto",
        "//mediapipe/tasks/cc/vision/interactive_segmenter:interactive_segmenter_graph",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/components/containers:MPPRegionOfInterest",
        "//mediapipe/tasks/ios/core:MPPTaskInfo",
        "//mediapipe/tasks/ios/vision/core:MPPImage",
        "//mediapipe/tasks/ios/vision/core:MPPVisionPacketCreator",
        "//mediapipe/tasks/ios/vision/core:MPPVisionTaskRunner",
        "//mediapipe/tasks/ios/vision/interactive_segmenter/utils:MPPInteractiveSegmenterOptionsHelpers",
        "//mediapipe/tasks/ios/vision/interactive_segmenter/utils:MPPInteractiveSegmenterResultHelpers",
        "//mediapipe/util:label_map_cc_proto",
        "//third_party/apple_frameworks:Foundation",
    ],
)
