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
    name = "MPPTextClassifierOptionsHelpers",
    srcs = ["sources/MPPTextClassifierOptions+Helpers.mm"],
    hdrs = ["sources/MPPTextClassifierOptions+Helpers.h"],
    deps = [
        "//mediapipe/tasks/c/text/text_classifier:text_classifier_lib",
        "//mediapipe/tasks/ios/components/processors:MPPClassifierOptions",
        "//mediapipe/tasks/ios/components/processors/utils:MPPClassifierOptionsHelpers",
        "//mediapipe/tasks/ios/core/utils:MPPBaseOptionsHelpers",
        "//mediapipe/tasks/ios/text/text_classifier:MPPTextClassifierOptions",
    ],
)

objc_library(
    name = "MPPTextClassifierResultHelpers",
    srcs = ["sources/MPPTextClassifierResult+Helpers.mm"],
    hdrs = ["sources/MPPTextClassifierResult+Helpers.h"],
    deps = [
        "//mediapipe/tasks/c/text/text_classifier:text_classifier_lib",
        "//mediapipe/tasks/ios/components/containers/utils:MPPClassificationResultHelpers",
        "//mediapipe/tasks/ios/text/text_classifier:MPPTextClassifierResult",
    ],
)
