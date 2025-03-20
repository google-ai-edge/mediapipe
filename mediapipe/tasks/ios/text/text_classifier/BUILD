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

package(default_visibility = [
    "//mediapipe/tasks:internal",
])

licenses(["notice"])

objc_library(
    name = "MPPTextClassifierOptions",
    srcs = ["sources/MPPTextClassifierOptions.m"],
    hdrs = ["sources/MPPTextClassifierOptions.h"],
    deps = [
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPTextClassifierResult",
    srcs = ["sources/MPPTextClassifierResult.m"],
    hdrs = ["sources/MPPTextClassifierResult.h"],
    deps = [
        "//mediapipe/tasks/ios/components/containers:MPPClassificationResult",
        "//mediapipe/tasks/ios/core:MPPTaskResult",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPTextClassifier",
    srcs = ["sources/MPPTextClassifier.mm"],
    hdrs = ["sources/MPPTextClassifier.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
        "-x objective-c++",
    ],
    module_name = "MPPTextClassifier",
    deps = [
        ":MPPTextClassifierOptions",
        ":MPPTextClassifierResult",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_cc_proto",
        "//mediapipe/tasks/cc/text/text_classifier:text_classifier_graph",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/core:MPPTaskInfo",
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//mediapipe/tasks/ios/core:MPPTextPacketCreator",
        "//mediapipe/tasks/ios/text/core:MPPTextTaskRunner",
        "//mediapipe/tasks/ios/text/text_classifier/utils:MPPTextClassifierOptionsHelpers",
        "//mediapipe/tasks/ios/text/text_classifier/utils:MPPTextClassifierResultHelpers",
        "//third_party/apple_frameworks:Foundation",
    ],
)
