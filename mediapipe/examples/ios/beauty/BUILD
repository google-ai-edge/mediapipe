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
    "ios_framework")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")
load(
    "//mediapipe/examples/ios:bundle_id.bzl",
    "BUNDLE_ID_PREFIX",
    "example_provisioning",
)

FRAMEWORK_HEADERS = [
    "MediaPipeController.h",
]

IOS_FAMILIES = [
    "iphone",
    "ipad",
]

licenses(["notice"])

MIN_IOS_VERSION = "11.0"

alias(
    name = "beauty",
    actual = "BeautyApp",
)

ios_application(
    name = "BeautyApp",
    app_icons = ["//mediapipe/examples/ios/common:AppIcon"],
    bundle_id = BUNDLE_ID_PREFIX + ".Beauty",
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
        ":BeautyAppLibrary",
        "@ios_opencv//:OpencvFramework",
    ],
)

ios_framework(
    name = "MediaPipeFramework",
    hdrs = FRAMEWORK_HEADERS,
    bundle_id = "dh.MediaPipeFramework",
    bundle_name = "MediaPipeFramework",
    families = IOS_FAMILIES,
    infoplists = [
        "//mediapipe/examples/ios/common:Info.plist",
    ],
    minimum_os_version = MIN_IOS_VERSION,
    visibility = ["//visibility:public"],
    deps = [
        ":MediaPipeLib",
        "@ios_opencv//:OpencvFramework",
    ],
)


objc_library(
    name = "MediaPipeLib",
    srcs = [
        "MediaPipeController.mm",
    ],
    hdrs = FRAMEWORK_HEADERS,
    copts = ["-std=c++17"],  # https://github.com/google/mediapipe/issues/2275#issuecomment-877145926
    data = [
        "//mediapipe/graphs/face_effect:face_effect_gpu.binarypb",
        "//mediapipe/modules/face_detection:face_detection_short_range.tflite",
        "//mediapipe/graphs/face_effect/data:axis.binarypb",
        "//mediapipe/graphs/face_effect/data:axis.pngblob",
        "//mediapipe/graphs/face_effect/data:facepaint.pngblob",
        "//mediapipe/graphs/face_effect/data:glasses.binarypb",
        "//mediapipe/graphs/face_effect/data:glasses.pngblob",
        "//mediapipe/modules/face_geometry/data:geometry_pipeline_metadata.binarypb",
        "//mediapipe/modules/face_geometry/data:geometry_pipeline_metadata_detection.binarypb",
        "//mediapipe/modules/face_geometry/data:geometry_pipeline_metadata_landmarks.binarypb",
        "//mediapipe/modules/face_landmark:face_landmark.tflite",
        "//mediapipe/graphs/beauty:beauty_mobile_gpu.binarypb",
        "//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite",
    ],
    deps = [
        "//mediapipe/objc:mediapipe_framework_ios",
        "//mediapipe/objc:mediapipe_input_sources_ios",
        "//mediapipe/calculators/core:packet_presence_calculator",
    ] + select({
        "//conditions:default": [
            "//mediapipe/framework/formats:matrix_data_cc_proto",
            "//mediapipe/graphs/face_effect:face_effect_gpu_deps",
            "//mediapipe/modules/face_geometry/protos:face_geometry_cc_proto",
            "//mediapipe/graphs/beauty:mobile_calculators",
            "//mediapipe/framework/formats:landmark_cc_proto",
        ],
    }),
)


objc_library(
    name = "BeautyAppLibrary",
    srcs = [
        "BeautyViewController.mm",
    ],
    hdrs = [
        "BeautyViewController.h",
    ],
    copts = ["-std=c++17"],
    data = [
        "//mediapipe/graphs/beauty:beauty_mobile_gpu.binarypb",
        "//mediapipe/modules/face_detection:face_detection_short_range.tflite",
        "//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite",
    ],
    deps = [
        "//mediapipe/examples/ios/common:CommonMediaPipeAppLibrary",
    ] + select({
        "//mediapipe:ios_i386": [],
        "//mediapipe:ios_x86_64": [],
        "//conditions:default": [
            "//mediapipe/graphs/beauty:mobile_calculators",
            "//mediapipe/framework/formats:landmark_cc_proto",
        ],
    }),
)
