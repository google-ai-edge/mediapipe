# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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
    name = "MPPTextClassifier",
    srcs = ["sources/MPPTextClassifier.mm"],
    hdrs = ["sources/MPPTextClassifier.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
    ],
    deps = [
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//mediapipe/tasks/ios/core:MPPTaskInfo",
        "//mediapipe/tasks/ios/text/core:MPPBaseTextTaskApi",
        "//mediapipe/tasks/ios/core:MPPPacketCreator",
        "//mediapipe/tasks/ios/text/text_classifier/utils:MPPTextClassifierOptionsHelpers",
        "//mediapipe/tasks/ios/components/containers/utils:MPPClassificationResultHelpers",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        ":MPPTextClassifierOptions",
    ],
)

objc_library(
    name = "MPPTextClassifierOptions",
    srcs = ["sources/MPPTextClassifierOptions.m"],
    hdrs = ["sources/MPPTextClassifierOptions.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
    ],
    deps = [
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//mediapipe/tasks/ios/components/processors:MPPClassifierOptions",
    ],
)

