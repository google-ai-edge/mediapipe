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

licenses(["notice"])  # Apache 2.0
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_framework")

ios_framework(
    name = "CommonMediaPipeFramework",
    hdrs = [
        "CommonLibraryFactory.h",
        "CommonViewController.h",
    ],
    infoplists = ["Info.plist"],
    bundle_id = "com.ola.olarender.common",
    families = ["iphone", "ipad"],
    minimum_os_version = "11.0",
    deps = [
        ":CommonMediaPipeAppLibrary",
        "@ios_opencv//:OpencvFramework",
    ] + select({
        "//mediapipe:ios_i386": [],
        "//mediapipe:ios_x86_64": [],
        "//conditions:default": [
            "//mediapipe/graphs/face_mesh:mobile_calculators",
            "//mediapipe/framework/formats:landmark_cc_proto",
        ],
    }),
)

objc_library(
    name = "CommonMediaPipeAppLibrary",
    srcs = [
       "AppDelegate.mm",
        "CommonViewController.mm",
        "CommonLibraryFactory.mm",
       "main.m",
    ],
    hdrs = [
       "AppDelegate.h",
        "CommonViewController.h",
        "CommonLibraryFactory.h",
    ],
   data = [
       "Base.lproj/LaunchScreen.storyboard",
       "Base.lproj/Main.storyboard",
   ],
    sdk_frameworks = [
        "AVFoundation",
        "CoreGraphics",
        "CoreMedia",
        "UIKit",
    ],
     copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    visibility = [
        "//mediapipe:__subpackages__",
    ],
    deps = [
        "//mediapipe/objc:mediapipe_framework_ios",
        "//mediapipe/objc:mediapipe_input_sources_ios",
        "//mediapipe/objc:mediapipe_layer_renderer",
    ],
)

exports_files(["Info.plist"])

filegroup(
    name = "AppIcon",
    srcs = glob(["Assets.xcassets/AppIcon.appiconset/*"]),
    visibility = ["//mediapipe:__subpackages__"],
)
