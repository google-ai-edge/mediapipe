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
    name = "MPPImageTestUtils",
    srcs = ["sources/MPPImage+TestUtils.mm"],
    hdrs = ["sources/MPPImage+TestUtils.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
        "-x objective-c++",
    ],
    module_name = "MPPImageTestUtils",
    deps = [
        "//mediapipe/tasks/ios/test/utils:MPPFileInfo",
        "//mediapipe/tasks/ios/vision/core:MPPImage",
    ],
)

objc_library(
    name = "MPPMaskTestUtils",
    srcs = ["sources/MPPMask+TestUtils.m"],
    hdrs = ["sources/MPPMask+TestUtils.h"],
    module_name = "MPPMaskTestUtils",
    deps = [
        "//mediapipe/tasks/ios/test/utils:MPPFileInfo",
        "//mediapipe/tasks/ios/vision/core:MPPMask",
        "//third_party/apple_frameworks:UIKit",
    ],
)

cc_library(
    name = "parse_proto_utils",
    srcs = ["sources/parse_proto_utils.cc"],
    hdrs = ["sources/parse_proto_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_protobuf//:protobuf",
    ],
)
