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

load(
    "//mediapipe/framework/tool:ios.bzl",
    "MPP_TASK_MINIMUM_OS_VERSION",
    "strip_api_include_path_prefix",
)
load("@build_bazel_rules_apple//apple:apple.bzl", "apple_static_xcframework")
load("@build_bazel_rules_apple//apple:apple_static_library.bzl", "apple_static_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# List of targets to be added in avoid_deps of ":MediaPipeTasksVision_framework"
# and ":MediaPipeTasksText_framework".
# The transitive closure of the following targets are used for building the
# frameworks but are avoided from the framework binaries to avoid duplicate symbols
# error when included in an xcode project:
# 1. iOS classes shared amongst the various vision and text tasks. These classes
# will be built with ":MediaPipeTasksCommon_framework"
# 2. Task graphs. These will be built with ":MediaPipeTaskGraphs_library".
# 3. gpu targets which will be built with the ":MediaPipeTaskGraphs_library".
#
# Instead of linking options and containers, we link their helpers to
# `MPPTasksCommon` to avoid duplicated method warnings in categories when text
# and vision frameworks are installed in the same Xcode project.
OBJC_TASK_COMMON_DEPS = [
    "//mediapipe/tasks/ios/core/utils:MPPBaseOptionsHelpers",
    "//mediapipe/tasks/ios/core:MPPTaskInfo",
    "//mediapipe/tasks/ios/core:MPPTaskOptions",
    "//mediapipe/tasks/ios/core:MPPTaskResult",
    "//mediapipe/tasks/ios/core:MPPTaskRunner",
    "//mediapipe/tasks/ios/components/containers/utils:MPPClassificationResultHelpers",
    "//mediapipe/tasks/ios/components/containers/utils:MPPCategoryHelpers",
    "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
]

CALCULATORS_AND_GRAPHS = [
    "//mediapipe/tasks/cc/vision/image_classifier:image_classifier_graph",
    "//mediapipe/tasks/cc/vision/object_detector:object_detector_graph",
    "//mediapipe/tasks/cc/text/text_classifier:text_classifier_graph",
    "//mediapipe/tasks/cc/text/text_embedder:text_embedder_graph",
    "//mediapipe/calculators/core:flow_limiter_calculator",
]

strip_api_include_path_prefix(
    name = "strip_api_include_path",
    hdr_labels = [
        "//mediapipe/tasks/ios/common:sources/MPPCommon.h",
        "//mediapipe/tasks/ios/components/containers:sources/MPPCategory.h",
        "//mediapipe/tasks/ios/components/containers:sources/MPPClassificationResult.h",
        "//mediapipe/tasks/ios/components/containers:sources/MPPEmbedding.h",
        "//mediapipe/tasks/ios/components/containers:sources/MPPEmbeddingResult.h",
        "//mediapipe/tasks/ios/components/containers:sources/MPPDetection.h",
        "//mediapipe/tasks/ios/core:sources/MPPBaseOptions.h",
        "//mediapipe/tasks/ios/core:sources/MPPTaskOptions.h",
        "//mediapipe/tasks/ios/core:sources/MPPTaskResult.h",
        "//mediapipe/tasks/ios/text/text_classifier:sources/MPPTextClassifier.h",
        "//mediapipe/tasks/ios/text/text_classifier:sources/MPPTextClassifierOptions.h",
        "//mediapipe/tasks/ios/text/text_classifier:sources/MPPTextClassifierResult.h",
        "//mediapipe/tasks/ios/text/text_embedder:sources/MPPTextEmbedder.h",
        "//mediapipe/tasks/ios/text/text_embedder:sources/MPPTextEmbedderOptions.h",
        "//mediapipe/tasks/ios/text/text_embedder:sources/MPPTextEmbedderResult.h",
        "//mediapipe/tasks/ios/vision/core:sources/MPPRunningMode.h",
        "//mediapipe/tasks/ios/vision/core:sources/MPPImage.h",
        "//mediapipe/tasks/ios/vision/image_classifier:sources/MPPImageClassifier.h",
        "//mediapipe/tasks/ios/vision/image_classifier:sources/MPPImageClassifierOptions.h",
        "//mediapipe/tasks/ios/vision/image_classifier:sources/MPPImageClassifierResult.h",
        "//mediapipe/tasks/ios/vision/object_detector:sources/MPPObjectDetector.h",
        "//mediapipe/tasks/ios/vision/object_detector:sources/MPPObjectDetectorOptions.h",
        "//mediapipe/tasks/ios/vision/object_detector:sources/MPPObjectDetectionResult.h",
    ],
)

apple_static_xcframework(
    name = "MediaPipeTasksText_framework",
    # Avoid dependencies of ":MediaPipeTasksCommon_framework" and
    # ":MediaPipeTaskGraphs_library in order to prevent duplicate symbols error
    # when the frameworks are imported in iOS projects.
    avoid_deps = OBJC_TASK_COMMON_DEPS + CALCULATORS_AND_GRAPHS + [
        "//mediapipe/gpu:metal_shared_resources",
    ],
    bundle_name = "MediaPipeTasksText",
    ios = {
        "simulator": [
            "arm64",
            "x86_64",
        ],
        "device": ["arm64"],
    },
    minimum_os_versions = {
        "ios": MPP_TASK_MINIMUM_OS_VERSION,
    },
    public_hdrs = [
        ":MPPBaseOptions.h",
        ":MPPCategory.h",
        ":MPPClassificationResult.h",
        ":MPPEmbedding.h",
        ":MPPEmbeddingResult.h",
        ":MPPCommon.h",
        ":MPPTaskOptions.h",
        ":MPPTaskResult.h",
        ":MPPTextClassifier.h",
        ":MPPTextClassifierOptions.h",
        ":MPPTextClassifierResult.h",
        ":MPPTextEmbedder.h",
        ":MPPTextEmbedderOptions.h",
        ":MPPTextEmbedderResult.h",
    ],
    deps = [
        "//mediapipe/tasks/ios/text/text_classifier:MPPTextClassifier",
        "//mediapipe/tasks/ios/text/text_embedder:MPPTextEmbedder",
    ],
)

apple_static_xcframework(
    name = "MediaPipeTasksVision_framework",
    # Avoids dependencies of ":MediaPipeTasksCommon_framework" and
    # ":MediaPipeTaskGraphs_library in order to prevent duplicate symbols error
    # when the frameworks are imported in iOS projects.
    # Also avoids opencv since it will be built with
    # ":MediaPipeTaskGraphs_library".
    avoid_deps = OBJC_TASK_COMMON_DEPS + CALCULATORS_AND_GRAPHS + [
        "//mediapipe/gpu:metal_shared_resources",
    ],
    bundle_name = "MediaPipeTasksVision",
    ios = {
        "simulator": [
            "arm64",
            "x86_64",
        ],
        "device": ["arm64"],
    },
    minimum_os_versions = {
        "ios": MPP_TASK_MINIMUM_OS_VERSION,
    },
    public_hdrs = [
        ":MPPBaseOptions.h",
        ":MPPCategory.h",
        ":MPPClassificationResult.h",
        ":MPPDetection.h",
        ":MPPCommon.h",
        ":MPPTaskOptions.h",
        ":MPPTaskResult.h",
        ":MPPImage.h",
        ":MPPRunningMode.h",
        ":MPPImageClassifier.h",
        ":MPPImageClassifierOptions.h",
        ":MPPImageClassifierResult.h",
        ":MPPObjectDetector.h",
        ":MPPObjectDetectorOptions.h",
        ":MPPObjectDetectionResult.h",
    ],
    deps = [
        "//mediapipe/tasks/ios/vision/image_classifier:MPPImageClassifier",
        "//mediapipe/tasks/ios/vision/object_detector:MPPObjectDetector",
    ],
)

apple_static_library(
    name = "MediaPipeTaskGraphs_library",
    # There is no way to turn off zlib dependency in custom opencv builds.
    # Hence zlib is avoided to prevent duplicate symbols because of conflicts
    # between opencv's zlib and "@zlib//:zlib"
    avoid_deps = [
        "@zlib//:zlib",
    ],
    minimum_os_version = MPP_TASK_MINIMUM_OS_VERSION,
    platform_type = "ios",
    deps = CALCULATORS_AND_GRAPHS + [
        "@org_tensorflow//third_party/icu/data:conversion_data",
    ] + select({
        "//third_party:opencv_ios_sim_arm64_source_build": ["@ios_opencv_source//:opencv_xcframework"],
        "//third_party:opencv_ios_arm64_source_build": ["@ios_opencv_source//:opencv_xcframework"],
        "//third_party:opencv_ios_x86_64_source_build": ["@ios_opencv_source//:opencv_xcframework"],
        "//third_party:opencv_ios_sim_fat_source_build": ["@ios_opencv_source//:opencv_xcframework"],
        "//conditions:default": ["@ios_opencv//:OpencvFramework"],
    }),
)

apple_static_xcframework(
    name = "MediaPipeTasksCommon_framework",
    # avoids gpu targets since they will be built with
    # ":MediaPipeTaskGraphs_library". Otherwise it will result in
    # duplicate symbols error when the frameworks are imported in iOS.
    avoid_deps = [
        "//mediapipe/gpu:metal_shared_resources",
    ],
    bundle_name = "MediaPipeTasksCommon",
    ios = {
        "simulator": [
            "arm64",
            "x86_64",
        ],
        "device": ["arm64"],
    },
    minimum_os_versions = {
        "ios": MPP_TASK_MINIMUM_OS_VERSION,
    },
    deps = OBJC_TASK_COMMON_DEPS,
)
