# Copyright 2020 The MediaPipe Authors.
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
    name = "faceeffect",
    actual = "FaceEffectApp",
)

ios_application(
    name = "FaceEffectApp",
    app_icons = ["//mediapipe/examples/ios/common:AppIcon"],
    bundle_id = BUNDLE_ID_PREFIX + ".FaceEffectGpu",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = ["Info.plist"],
    minimum_os_version = MIN_IOS_VERSION,
    provisioning_profile = example_provisioning(),
    deps = [
        ":FaceEffectAppLibrary",
        "@ios_opencv//:OpencvFramework",
    ],
)

objc_library(
    name = "FaceEffectViewController",
    srcs = [
        "FaceEffectViewController.mm",
    ],
    hdrs = [
        "FaceEffectViewController.h",
    ],
    data = [
        "Base.lproj/LaunchScreen.storyboard",
        "Base.lproj/Main.storyboard",
        "//mediapipe/graphs/face_effect:face_effect_gpu.binarypb",
        "//mediapipe/graphs/face_effect/data:axis.binarypb",
        "//mediapipe/graphs/face_effect/data:axis.pngblob",
        "//mediapipe/graphs/face_effect/data:facepaint.pngblob",
        "//mediapipe/graphs/face_effect/data:glasses.binarypb",
        "//mediapipe/graphs/face_effect/data:glasses.pngblob",
        "//mediapipe/modules/face_detection:face_detection_short_range.tflite",
        "//mediapipe/modules/face_geometry/data:geometry_pipeline_metadata.binarypb",
        "//mediapipe/modules/face_geometry/data:geometry_pipeline_metadata_detection.binarypb",
        "//mediapipe/modules/face_geometry/data:geometry_pipeline_metadata_landmarks.binarypb",
        "//mediapipe/modules/face_landmark:face_landmark.tflite",
    ],
    features = ["-layering_check"],
    deps = [
        "//mediapipe/framework/formats:matrix_data_cc_proto",
        "//mediapipe/modules/face_geometry/protos:face_geometry_cc_proto",
        "//mediapipe/objc:mediapipe_framework_ios",
        "//mediapipe/objc:mediapipe_input_sources_ios",
        "//mediapipe/objc:mediapipe_layer_renderer",
        "//third_party/apple_frameworks:AVFoundation",
        "//third_party/apple_frameworks:CoreGraphics",
        "//third_party/apple_frameworks:CoreMedia",
        "//third_party/apple_frameworks:UIKit",
    ] + select({
        "//mediapipe:ios_x86_64": [],
        "//conditions:default": [
            "//mediapipe/graphs/face_effect:face_effect_gpu_deps",
        ],
    }),
)

objc_library(
    name = "FaceEffectAppLibrary",
    srcs = [
        "AppDelegate.m",
        "main.m",
    ],
    hdrs = [
        "AppDelegate.h",
    ],
    data = [
        "Base.lproj/LaunchScreen.storyboard",
        "Base.lproj/Main.storyboard",
        "//mediapipe/graphs/face_effect:face_effect_gpu.binarypb",
        "//mediapipe/graphs/face_effect/data:facepaint.pngblob",
        "//mediapipe/graphs/face_effect/data:glasses.binarypb",
        "//mediapipe/graphs/face_effect/data:glasses.pngblob",
        "//mediapipe/modules/face_detection:face_detection_short_range.tflite",
        "//mediapipe/modules/face_geometry/data:geometry_pipeline_metadata.binarypb",
        "//mediapipe/modules/face_landmark:face_landmark.tflite",
    ],
    deps = [
        ":FaceEffectViewController",
    ],
)
