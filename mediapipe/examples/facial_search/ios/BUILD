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

load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application", "ios_framework")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")
load(
    "//mediapipe/examples/ios:bundle_id.bzl",
    "BUNDLE_ID_PREFIX",
    "example_provisioning",
)

licenses(["notice"])  # Apache 2.0

MIN_IOS_VERSION = "11.1"

IOS_FAMILIES = [
    "iphone",
    "ipad",
]

# Build this .ipa with:
# bazel build --config=ios_arm64 mediapipe/examples/facial_search/ios:FacialSearchGpuApp
ios_application(
    name = "FacialSearchGpuApp",
    app_icons = ["//mediapipe/examples/ios/common:AppIcon"],
    bundle_id = BUNDLE_ID_PREFIX + ".FacialSearchGpu",
    families = IOS_FAMILIES,
    infoplists = [
        "//mediapipe/examples/ios/common:Info.plist",
        "Info.plist",
    ],
    minimum_os_version = MIN_IOS_VERSION,
    provisioning_profile = example_provisioning(),
    deps = [":FacialSearchGpuAppLibrary"],
)

swift_library(
    name = "FacialSearchGpuAppLibrary",
    srcs = [
        "AppDelegate.swift",
        "Cameras.swift",
        "DotDot.swift",
        "FacialSearchViewController.swift",
    ],
    data = ["Base.lproj/Main.storyboard"],
    deps = [
        ":ObjcppLib",
        "@ios_opencv//:OpencvFramework",
    ],
)

FRAMEWORK_HEADERS = [
    "ObjcppLib.h",
]

# Using Mediapipe in an iOS Swift app (without Tulsi)
# Create new Swift app: XCode: File > New > Project...
# Select iOS > "Single View App" > Next
# Select Language: Swift > Next > Create
# Delete these files from the new project: (Move to Trash)
# * AppDelegate.swift
# * ViewController.swift
# Copy these files to your app: (if asked, do not create a bridging header)
# * AppDelegate.swift
# * Cameras.swift
# * DotDot.swift
# * FacialSearchViewController.swift
# Edit your app's Info.plist:
# * Create key "NSCameraUsageDescription" with value: "This app uses the camera to demonstrate live video processing."
# Edit your Main.storyboard's custom class, setting it to FacialSearchViewController (in the Identity inspector)
# Run: bazel build --copt=-fembed-bitcode --apple_bitcode=embedded --config=ios_arm64 mediapipe/examples/facial_search/ios:FacialSearch
# (some linker warnings about global C++ symbols may appear)
# Run: ./mediapipe/examples/facial_search/ios/patch_ios_framework.sh bazel-bin/mediapipe/examples/facial_search/ios/FacialSearch.zip ObjcppLib.h
# Note: append the contents of FRAMEWORK_HEADERS separated by spaces (here: ObjcppLib.h).
# Run: open bazel-bin/mediapipe/examples/facial_search/ios and drag and drop
# the FacialSearch.framework folder into your app files (check: Copy items if needed > Finish)
# Make sure the framework gets embedded into the app:
# In General > Frameworks, Libraries, and Embedded Content set FacialSearch.framework to "Embed & Sign".
# Connect your device and run.
ios_framework(
    name = "FacialSearch",
    hdrs = FRAMEWORK_HEADERS,
    bundle_id = BUNDLE_ID_PREFIX + ".FacialSearchFramework",
    bundle_name = "FacialSearch",
    families = IOS_FAMILIES,
    infoplists = [
        "//mediapipe/examples/ios/common:Info.plist",
        "Info.plist",
    ],
    minimum_os_version = MIN_IOS_VERSION,
    visibility = ["//visibility:public"],
    deps = [
        ":ObjcppLib",
        "@ios_opencv//:OpencvFramework",
    ],
)

objc_library(
    name = "ObjcppLib",
    srcs = [
        "Classification.mm",
        "ObjcppLib.mm",
    ],
    hdrs = FRAMEWORK_HEADERS,
    copts = ["-std=c++17"],  # https://github.com/google/mediapipe/issues/2275#issuecomment-877145926
    data = [
        "//mediapipe/examples/facial_search/graphs:facial_search_gpu.binarypb",
        "//mediapipe/examples/facial_search/images:jpgs",
        "//mediapipe/models:face_embeddings.tflite",
        "//mediapipe/modules/face_detection:face_detection_short_range.tflite",
    ],
    deps = [
        "//mediapipe/objc:mediapipe_framework_ios",
        "//mediapipe/objc:mediapipe_input_sources_ios",
        "//mediapipe/objc:mediapipe_layer_renderer",
    ] + select({
        "//mediapipe:ios_i386": [],
        "//mediapipe:ios_x86_64": [],
        "//conditions:default": [
            "//mediapipe/examples/facial_search/graphs:gpu_calculators",
            "//mediapipe/examples/facial_search:embeddings_database",
        ],
    }),
)
