# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
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
    name = "MPPTextClassifierOptions",
    srcs = ["sources/MPPTextClassifierOptions.m"],
    hdrs = ["sources/MPPTextClassifierOptions.h"],
    deps = [
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//mediapipe/tasks/ios/components/processors:MPPClassifierOptions",
    ],
)

objc_library(
    name = "MPPTextClassifierResult",
    srcs = ["sources/MPPTextClassifierResult.m"],
    hdrs = ["sources/MPPTextClassifierResult.h"],
    deps = [
        "//mediapipe/tasks/ios/core:MPPTaskResult",
        "//mediapipe/tasks/ios/components/containers:MPPClassificationResult",
    ],
)
