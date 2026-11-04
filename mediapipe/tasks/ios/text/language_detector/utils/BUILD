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
    name = "MPPLanguageDetectorOptionsHelpers",
    srcs = ["sources/MPPLanguageDetectorOptions+Helpers.mm"],
    hdrs = ["sources/MPPLanguageDetectorOptions+Helpers.h"],
    deps = [
        "//mediapipe/tasks/c/text/language_detector:language_detector_lib",
        "//mediapipe/tasks/ios/components/processors:MPPClassifierOptions",
        "//mediapipe/tasks/ios/components/processors/utils:MPPClassifierOptionsHelpers",
        "//mediapipe/tasks/ios/core/utils:MPPBaseOptionsHelpers",
        "//mediapipe/tasks/ios/text/language_detector:MPPLanguageDetectorOptions",
    ],
)

objc_library(
    name = "MPPLanguageDetectorResultHelpers",
    srcs = ["sources/MPPLanguageDetectorResult+Helpers.mm"],
    hdrs = ["sources/MPPLanguageDetectorResult+Helpers.h"],
    deps = [
        "//mediapipe/tasks/c/text/language_detector:language_detector_lib",
        "//mediapipe/tasks/ios/text/language_detector:MPPLanguageDetectorResult",
    ],
)
