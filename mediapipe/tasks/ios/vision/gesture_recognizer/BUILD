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
    name = "MPPGestureRecognizerResult",
    srcs = ["sources/MPPGestureRecognizerResult.m"],
    hdrs = ["sources/MPPGestureRecognizerResult.h"],
    deps = [
        "//mediapipe/tasks/ios/components/containers:MPPCategory",
        "//mediapipe/tasks/ios/components/containers:MPPLandmark",
        "//mediapipe/tasks/ios/core:MPPTaskResult",
    ],
)

objc_library(
    name = "MPPGestureRecognizerOptions",
    srcs = ["sources/MPPGestureRecognizerOptions.m"],
    hdrs = ["sources/MPPGestureRecognizerOptions.h"],
    deps = [
        ":MPPGestureRecognizerResult",
        "//mediapipe/tasks/ios/components/processors:MPPClassifierOptions",
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//mediapipe/tasks/ios/vision/core:MPPRunningMode",
    ],
)
