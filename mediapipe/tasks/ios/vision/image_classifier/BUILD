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

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

objc_library(
    name = "MPPImageClassifierResult",
    srcs = ["sources/MPPImageClassifierResult.m"],
    hdrs = ["sources/MPPImageClassifierResult.h"],
    deps = [
        "//mediapipe/tasks/ios/components/containers:MPPClassificationResult",
        "//mediapipe/tasks/ios/core:MPPTaskResult",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPImageClassifierOptions",
    srcs = ["sources/MPPImageClassifierOptions.m"],
    hdrs = ["sources/MPPImageClassifierOptions.h"],
    deps = [
        ":MPPImageClassifierResult",
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//mediapipe/tasks/ios/vision/core:MPPRunningMode",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPImageClassifier",
    srcs = ["sources/MPPImageClassifier.mm"],
    hdrs = ["sources/MPPImageClassifier.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
        "-x objective-c++",
    ],
    module_name = "MPPImageClassifier",
    deps = [
        ":MPPImageClassifierOptions",
        ":MPPImageClassifierResult",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_cc_proto",
        "//mediapipe/tasks/cc/vision/image_classifier:image_classifier_graph",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/core:MPPTaskInfo",
        "//mediapipe/tasks/ios/vision/core:MPPImage",
        "//mediapipe/tasks/ios/vision/core:MPPVisionPacketCreator",
        "//mediapipe/tasks/ios/vision/core:MPPVisionTaskRunner",
        "//mediapipe/tasks/ios/vision/image_classifier/utils:MPPImageClassifierOptionsHelpers",
        "//mediapipe/tasks/ios/vision/image_classifier/utils:MPPImageClassifierResultHelpers",
        "//third_party/apple_frameworks:Foundation",
    ],
)
