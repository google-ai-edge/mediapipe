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
# python3 mediapipe/examples/ios/link_local_profiles.py
#bazel build -c opt --config=ios_arm64 mediapipe/examples/ios/PoseLandmarkerApp:PoseLandmarkerApp

load(
   "//mediapipe/framework/tool:ios.bzl",
   "MPP_TASK_MINIMUM_OS_VERSION",
   "strip_api_include_path_prefix",
)

load(
    "@build_bazel_rules_apple//apple:ios.bzl",
    "ios_framework",
)

load(
    "//mediapipe/examples/ios:bundle_id.bzl",
    "BUNDLE_ID_PREFIX",
    "example_provisioning",
)

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

MIN_IOS_VERSION = MPP_TASK_MINIMUM_OS_VERSION

alias(
    name = "poselandmarkerapp",
    actual = "PoseLandmarkerApp",
)

OBJC_TASK_COMMON_DEPS = [
    "//mediapipe/tasks/ios/core/utils:MPPBaseOptionsHelpers",
    "//mediapipe/tasks/ios/core:MPPTaskInfo",
    "//mediapipe/tasks/ios/core:MPPTaskOptions",
    "//mediapipe/tasks/ios/core:MPPTaskResult",
    "//mediapipe/tasks/ios/core:MPPTaskRunner",
    "//mediapipe/tasks/ios/components/containers/utils:MPPClassificationResultHelpers",
    "//mediapipe/tasks/ios/components/containers/utils:MPPCategoryHelpers",
    "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
    "//mediapipe/tasks/ios/vision/core:MPPMask",
    "//mediapipe/tasks/ios/components/containers:MPPLandmark",
    "//mediapipe/tasks/ios/components/containers:MPPConnection",
    "//mediapipe/tasks/ios/components/containers:MPPDetection",
    "//mediapipe/tasks/ios/components/containers:MPPEmbeddingResult",
    "//mediapipe/tasks/ios/components/containers:MPPEmbedding",
    "//mediapipe/tasks/ios/components/containers:MPPClassificationResult",
    "//mediapipe/tasks/ios/components/containers:MPPCategory",
]

CALCULATORS_AND_GRAPHS = [
    "//mediapipe/calculators/core:flow_limiter_calculator",
    "//mediapipe/tasks/cc/vision/pose_landmarker:pose_landmarker_graph",
]

strip_api_include_path_prefix(
    name = "strip_api_include_path",
    hdr_labels = [
        "//mediapipe/tasks/ios/common:sources/MPPCommon.h",
        "//mediapipe/tasks/ios/components/containers:sources/MPPCategory.h",
        "//mediapipe/tasks/ios/components/containers:sources/MPPClassificationResult.h",
        "//mediapipe/tasks/ios/components/containers:sources/MPPEmbedding.h",
        "//mediapipe/tasks/ios/components/containers:sources/MPPEmbeddingResult.h",
        "//mediapipe/tasks/ios/components/containers:sources/MPPConnection.h",
        "//mediapipe/tasks/ios/components/containers:sources/MPPDetection.h",
        "//mediapipe/tasks/ios/components/containers:sources/MPPLandmark.h",
        "//mediapipe/tasks/ios/vision/core:sources/MPPMask.h",
        "//mediapipe/tasks/ios/core:sources/MPPBaseOptions.h",
        "//mediapipe/tasks/ios/core:sources/MPPTaskOptions.h",
        "//mediapipe/tasks/ios/core:sources/MPPTaskResult.h",
        "//mediapipe/tasks/ios/vision/core:sources/MPPRunningMode.h",
        "//mediapipe/tasks/ios/vision/core:sources/MPPImage.h",
        "//mediapipe/tasks/ios/vision/pose_landmarker:sources/MPPPoseLandmarker.h",
        "//mediapipe/tasks/ios/vision/pose_landmarker:sources/MPPPoseLandmarkerOptions.h",
        "//mediapipe/tasks/ios/vision/pose_landmarker:sources/MPPPoseLandmarkerResult.h",
    ],
)

ios_framework(
    name = "PoseLandmarkerApp",
    hdrs = [
        "PoseLandmarkerSDK.h",
        ":MPPBaseOptions.h",
        ":MPPCategory.h",
        ":MPPClassificationResult.h",
        ":MPPDetection.h",
        ":MPPLandmark.h",
        ":MPPConnection.h",
        ":MPPCommon.h",
        ":MPPTaskOptions.h",
        ":MPPTaskResult.h",
        ":MPPImage.h",
        ":MPPRunningMode.h",
        ":MPPPoseLandmarker.h",
        ":MPPPoseLandmarkerOptions.h",
        ":MPPPoseLandmarkerResult.h",
        ":MPPMask.h",
    ],
    bundle_id = "com.google.mediapipe.poselandmarkersdk",
    bundle_name = "PoseLandmarkerSDK",
    families = [
        "iphone",
        "ipad",
    ],
    resources = [
        "Modules"
    ],
    infoplists = [
        "Info.plist",
    ],
    minimum_os_version = MIN_IOS_VERSION,
    visibility = ["//visibility:public"],
    deps = OBJC_TASK_COMMON_DEPS + CALCULATORS_AND_GRAPHS + [
        "@ios_opencv//:OpencvFramework",
        "@org_tensorflow//third_party/icu/data:conversion_data",
        "//mediapipe/tasks/ios/vision/pose_landmarker:MPPPoseLandmarker",
    ],
)

