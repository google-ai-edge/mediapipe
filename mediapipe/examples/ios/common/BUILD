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

licenses(["notice"])

objc_library(
    name = "CommonMediaPipeAppLibrary",
    srcs = [
        "AppDelegate.mm",
        "CommonViewController.mm",
        "main.m",
    ],
    hdrs = [
        "AppDelegate.h",
        "CommonViewController.h",
    ],
    data = [
        "Base.lproj/LaunchScreen.storyboard",
        "Base.lproj/Main.storyboard",
    ],
    visibility = [
        "//mediapipe:__subpackages__",
    ],
    deps = [
        "//mediapipe/objc:mediapipe_framework_ios",
        "//mediapipe/objc:mediapipe_input_sources_ios",
        "//mediapipe/objc:mediapipe_layer_renderer",
        "//third_party/apple_frameworks:AVFoundation",
        "//third_party/apple_frameworks:CoreGraphics",
        "//third_party/apple_frameworks:CoreMedia",
        "//third_party/apple_frameworks:UIKit",
    ],
)

exports_files(["Info.plist"])

filegroup(
    name = "AppIcon",
    srcs = glob(["Assets.xcassets/AppIcon.appiconset/*"]),
    visibility = ["//mediapipe:__subpackages__"],
)
