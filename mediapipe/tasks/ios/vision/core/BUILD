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
    name = "MPPImage",
    srcs = ["sources/MPPImage.m"],
    hdrs = ["sources/MPPImage.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
    ],
    module_name = "MPPImage",
    deps = [
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//third_party/apple_frameworks:CoreMedia",
        "//third_party/apple_frameworks:CoreVideo",
        "//third_party/apple_frameworks:UIKit",
    ],
)

objc_library(
    name = "MPPRunningMode",
    hdrs = ["sources/MPPRunningMode.h"],
    module_name = "MPPRunningMode",
)

objc_library(
    name = "MPPVisionPacketCreator",
    srcs = ["sources/MPPVisionPacketCreator.mm"],
    hdrs = ["sources/MPPVisionPacketCreator.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
    ],
    deps = [
        ":MPPImage",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/tasks/ios/components/containers:MPPRegionOfInterest",
        "//mediapipe/tasks/ios/components/containers/utils:MPPRegionOfInterestHelpers",
        "//mediapipe/tasks/ios/vision/core/utils:MPPImageUtils",
    ],
)

objc_library(
    name = "MPPVisionTaskRunner",
    srcs = ["sources/MPPVisionTaskRunner.mm"],
    hdrs = ["sources/MPPVisionTaskRunner.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
    ],
    deps = [
        ":MPPImage",
        ":MPPRunningMode",
        ":MPPVisionPacketCreator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/framework:packet",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/core:MPPTaskInfo",
        "//mediapipe/tasks/ios/core:MPPTaskRunner",
        "//third_party/apple_frameworks:UIKit",
        "@com_google_absl//absl/status:statusor",
    ],
)

objc_library(
    name = "MPPMask",
    srcs = ["sources/MPPMask.mm"],
    hdrs = ["sources/MPPMask.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
    ],
    deps = [
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
    ],
)
