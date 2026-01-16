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
    name = "MPPLanguageDetectorOptions",
    srcs = ["sources/MPPLanguageDetectorOptions.m"],
    hdrs = ["sources/MPPLanguageDetectorOptions.h"],
    deps = [
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPLanguageDetectorResult",
    srcs = ["sources/MPPLanguageDetectorResult.m"],
    hdrs = ["sources/MPPLanguageDetectorResult.h"],
    deps = [
        "//mediapipe/tasks/ios/core:MPPTaskResult",
    ],
)

objc_library(
    name = "MPPLanguageDetector",
    srcs = ["sources/MPPLanguageDetector.mm"],
    hdrs = ["sources/MPPLanguageDetector.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
        "-x objective-c++",
    ],
    module_name = "MPPLanguageDetector",
    deps = [
        ":MPPLanguageDetectorOptions",
        ":MPPLanguageDetectorResult",
        "//mediapipe/tasks/cc/text/text_classifier:text_classifier_graph",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/core:MPPTaskInfo",
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//mediapipe/tasks/ios/core:MPPTextPacketCreator",
        "//mediapipe/tasks/ios/text/core:MPPTextTaskRunner",
        "//mediapipe/tasks/ios/text/language_detector/utils:MPPLanguageDetectorOptionsHelpers",
        "//mediapipe/tasks/ios/text/language_detector/utils:MPPLanguageDetectorResultHelpers",
        "//third_party/apple_frameworks:Foundation",
    ],
)
