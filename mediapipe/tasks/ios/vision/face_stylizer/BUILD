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
    name = "MPPFaceStylizerResult",
    srcs = ["sources/MPPFaceStylizerResult.m"],
    hdrs = ["sources/MPPFaceStylizerResult.h"],
    deps = [
        "//mediapipe/tasks/ios/core:MPPTaskResult",
        "//mediapipe/tasks/ios/vision/core:MPPImage",
    ],
)

objc_library(
    name = "MPPFaceStylizerOptions",
    srcs = ["sources/MPPFaceStylizerOptions.m"],
    hdrs = ["sources/MPPFaceStylizerOptions.h"],
    deps = ["//mediapipe/tasks/ios/core:MPPTaskOptions"],
)

objc_library(
    name = "MPPFaceStylizer",
    srcs = ["sources/MPPFaceStylizer.mm"],
    hdrs = ["sources/MPPFaceStylizer.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
        "-x objective-c++",
        "",
    ],
    module_name = "MPPFaceStylizer",
    deps = [
        ":MPPFaceStylizerOptions",
        ":MPPFaceStylizerResult",
        "//mediapipe/tasks/cc/vision/face_stylizer:face_stylizer_graph",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/core:MPPTaskInfo",
        "//mediapipe/tasks/ios/vision/core:MPPImage",
        "//mediapipe/tasks/ios/vision/core:MPPVisionTaskRunner",
        "//mediapipe/tasks/ios/vision/face_stylizer/utils:MPPFaceStylizerOptionsHelpers",
        "//mediapipe/tasks/ios/vision/face_stylizer/utils:MPPFaceStylizerResultHelpers",
    ],
)
