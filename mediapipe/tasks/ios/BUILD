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

load("@rules_cc//cc:objc_library.bzl", "objc_library")
load(
    ":ios_build_rules.bzl",
    "mediapipe_static_xcframework",
)

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files([
    "MediaPipeTasksAudio.podspec.template",
    "MediaPipeTasksVision.podspec.template",
    "MediaPipeTasksText.podspec.template",
    "MediaPipeTasksRetrieval.podspec.template",
    "MediaPipeTasksCommon.podspec.template",
    "MediaPipeTasksCommon.swift.template",
    "PrivacyInfo.xcprivacy.template",
    "Package.swift.template",
    "MediaPipeTasksSPMSmokeTest.swift.template",
])

TENSORFLOW_LITE_C_DEPS = [
    "@litert//tflite:builtin_ops",
    "@litert//tflite/c:c_api",
    "@litert//tflite/c:c_api_experimental",
    "@litert//tflite/c:c_api_types",
    "@litert//tflite/c:common",
    "@litert//tflite/delegates/xnnpack:xnnpack_delegate",
    "@litert//tflite/delegates/gpu:metal_delegate",
    "@litert//tflite/delegates/gpu:metal_delegate_internal",
]

# MediaPipe iOS OpenCV build rules
config_setting(
    name = "opencv_ios_arm64_source_build",
    define_values = {
        "OPENCV": "source",
    },
    values = {
        "apple_platform_type": "ios",
        "ios_multi_cpus": "arm64",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "opencv_ios_sim_arm64_source_build",
    define_values = {
        "OPENCV": "source",
    },
    values = {
        "apple_platform_type": "ios",
        "ios_multi_cpus": "sim_arm64",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "opencv_ios_x86_64_source_build",
    define_values = {
        "OPENCV": "source",
    },
    values = {
        "apple_platform_type": "ios",
        "ios_multi_cpus": "x86_64",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "opencv_ios_sim_fat_source_build",
    define_values = {
        "OPENCV": "source",
    },
    values = {
        "apple_platform_type": "ios",
        "ios_multi_cpus": "sim_arm64,x86_64",
    },
    visibility = ["//visibility:public"],
)

# Targets here are force loaded to avoid stripping.
CALCULATORS_AND_GRAPHS = [
    "//mediapipe/calculators/core:flow_limiter_calculator",
    "//mediapipe/tasks/cc/audio/audio_classifier:audio_classifier_graph",
    "//mediapipe/tasks/cc/text/text_classifier:text_classifier_graph",
    "//mediapipe/tasks/cc/text/text_embedder:text_embedder_graph",
    "//mediapipe/tasks/cc/vision/face_detector:face_detector_graph",
    "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker_graph",
    "//mediapipe/tasks/cc/vision/hand_landmarker:hand_landmarker_graph",
    "//mediapipe/tasks/cc/vision/gesture_recognizer:gesture_recognizer_graph",
    "//mediapipe/tasks/cc/vision/image_classifier:image_classifier_graph",
    "//mediapipe/tasks/cc/vision/image_embedder:image_embedder_graph",
    "//mediapipe/tasks/cc/vision/image_segmenter:image_segmenter_graph",
    "//mediapipe/tasks/cc/vision/interactive_segmenter:interactive_segmenter_graph",
    "//mediapipe/tasks/cc/vision/interactive_segmenter_legacy:interactive_segmenter_legacy_graph",
    "//mediapipe/tasks/cc/vision/object_detector:object_detector_graph",
    "//mediapipe/tasks/cc/vision/pose_landmarker:pose_landmarker_graph",
    "//mediapipe/tasks/cc/vision/holistic_landmarker:holistic_landmarker_graph",
    # On Apple platforms this also statically links the LiteRT Metal GPU
    # accelerator, which self-registers and must be force loaded. See
    # //mediapipe/calculators/tensor:inference_calculator_litert.
    "//mediapipe/calculators/tensor:inference_calculator_litert",
    # LiteRT-LM CPU engine uses static self-registration that gets stripped unless force loaded.
    "@litert_lm//c:engine_cpu",
]

objc_library(
    name = "OpenCV",
    deps = select({
        ":opencv_ios_sim_arm64_source_build": [
            "//third_party:opencv",
        ],
        ":opencv_ios_arm64_source_build": [
            "//third_party:opencv",
        ],
        ":opencv_ios_x86_64_source_build": [
            "//third_party:opencv",
        ],
        ":opencv_ios_sim_fat_source_build": [
            "//third_party:opencv",
        ],
        "//conditions:default": [
            "//third_party:opencv",
        ],
    }),
    alwayslink = 1,
)

MEDIAPIPE_TASKS_COMMON_DEPS = TENSORFLOW_LITE_C_DEPS + [
    ":OpenCV",
    "//mediapipe/gpu:metal_shared_resources",
]

objc_library(
    name = "MediaPipeTaskGraphs",
    deps = CALCULATORS_AND_GRAPHS,
    alwayslink = 1,
)

mediapipe_static_xcframework(
    name = "MediaPipeTaskGraphs_library",
    bundle_name = "MediaPipeTaskGraphs_library",
    strip_rust_metadata = True,
    deps = [":MediaPipeTaskGraphs"],
)

mediapipe_static_xcframework(
    name = "MediaPipeTasksCommon_framework",
    avoid_deps = CALCULATORS_AND_GRAPHS,
    bundle_format = "framework",
    bundle_id = "com.google.mediapipe.tasks.common",
    bundle_name = "MediaPipeTasksCommon",
    infoplists = [":empty.plist"],
    # Including `OpenCV` and `TensorFlowLiteC` deps of the graphs here to avoid
    # conflicts with `TensorFlowLiteSwift` and `OpenCV` frameworks as iOS
    # `xcframeworks` handle duplicate dependencies with other iOS
    # `xcframeworks`. Including them with task graphs causes duplicate symbols
    # when installed alongside the respective frameworks since task graphs are
    # built as static libraries and force loaded.
    deps = MEDIAPIPE_TASKS_COMMON_DEPS,
)
