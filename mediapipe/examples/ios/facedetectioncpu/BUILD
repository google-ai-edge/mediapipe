# Copyright 2019 The MediaPipe Authors.
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

load(
    "@build_bazel_rules_apple//apple:ios.bzl",
    "ios_application",
)
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load(
    "//mediapipe/examples/ios:bundle_id.bzl",
    "BUNDLE_ID_PREFIX",
    "example_provisioning",
)

licenses(["notice"])

MIN_IOS_VERSION = "12.0"

alias(
    name = "facedetectioncpu",
    actual = "FaceDetectionCpuApp",
)

ios_application(
    name = "FaceDetectionCpuApp",
    app_icons = ["//mediapipe/examples/ios/common:AppIcon"],
    bundle_id = BUNDLE_ID_PREFIX + ".FaceDetectionCpu",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = [
        "//mediapipe/examples/ios/common:Info.plist",
        "Info.plist",
    ],
    minimum_os_version = MIN_IOS_VERSION,
    provisioning_profile = example_provisioning(),
    deps = [
        ":FaceDetectionCpuAppLibrary",
        "@ios_opencv//:OpencvFramework",
    ],
)

objc_library(
    name = "FaceDetectionCpuAppLibrary",
    data = [
        "//mediapipe/graphs/face_detection:face_detection_mobile_cpu.binarypb",
        "//mediapipe/modules/face_detection:face_detection_short_range.tflite",
    ],
    deps = [
        "//mediapipe/examples/ios/common:CommonMediaPipeAppLibrary",
    ] + select({
        "//mediapipe:ios_x86_64": [],
        "//conditions:default": [
            "//mediapipe/graphs/face_detection:mobile_calculators",
        ],
    }),
)
