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
    "//mediapipe/framework/tool:ios.bzl",
    "strip_api_include_path_prefix",
)
load(":ios_build_rules.bzl", "mediapipe_static_xcframework")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

TENSORFLOW_LITE_C_DEPS = [
    "@org_tensorflow//tensorflow/lite:builtin_ops",
    "@org_tensorflow//tensorflow/lite/c:c_api",
    "@org_tensorflow//tensorflow/lite/c:c_api_experimental",
    "@org_tensorflow//tensorflow/lite/c:c_api_types",
    "@org_tensorflow//tensorflow/lite/c:common",
    "@org_tensorflow//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
    "@org_tensorflow//tensorflow/lite/delegates/gpu:metal_delegate",
    "@org_tensorflow//tensorflow/lite/delegates/gpu:metal_delegate_internal",
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
    "//mediapipe/tasks/cc/vision/object_detector:object_detector_graph",
    "//mediapipe/tasks/cc/vision/pose_landmarker:pose_landmarker_graph",
    "//mediapipe/tasks/cc/vision/holistic_landmarker:holistic_landmarker_graph",
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
    "//mediapipe/tasks/ios/audio/audio_classifier:MPPAudioClassifier",
    "//mediapipe/tasks/ios/text/language_detector:MPPLanguageDetector",
    "//mediapipe/tasks/ios/text/text_classifier:MPPTextClassifier",
    "//mediapipe/tasks/ios/text/text_embedder:MPPTextEmbedder",
    "//mediapipe/tasks/ios/vision/face_detector:MPPFaceDetector",
    "//mediapipe/tasks/ios/vision/face_landmarker:MPPFaceLandmarker",
    "//mediapipe/tasks/ios/vision/gesture_recognizer:MPPGestureRecognizer",
    "//mediapipe/tasks/ios/vision/hand_landmarker:MPPHandLandmarker",
    "//mediapipe/tasks/ios/vision/holistic_landmarker:MPPHolisticLandmarker",
    "//mediapipe/tasks/ios/vision/image_classifier:MPPImageClassifier",
    "//mediapipe/tasks/ios/vision/image_embedder:MPPImageEmbedder",
    "//mediapipe/tasks/ios/vision/image_segmenter:MPPImageSegmenter",
    "//mediapipe/tasks/ios/vision/interactive_segmenter:MPPInteractiveSegmenter",
    "//mediapipe/tasks/ios/vision/object_detector:MPPObjectDetector",
    "//mediapipe/tasks/ios/vision/pose_landmarker:MPPPoseLandmarker",
]

objc_library(
    name = "MediaPipeTaskGraphs",
    deps = CALCULATORS_AND_GRAPHS,
    alwayslink = 1,
)

strip_api_include_path_prefix(
    name = "strip_api_include_path",
    hdr_labels = [
        "//mediapipe/tasks/ios/common:sources/MPPCommon.h",
        "//mediapipe/tasks/ios/components/containers:sources/MPPCategory.h",
        "//mediapipe/tasks/ios/components/containers:sources/MPPClassificationResult.h",
        "//mediapipe/tasks/ios/components/containers:sources/MPPConnection.h",
        "//mediapipe/tasks/ios/components/containers:sources/MPPDetection.h",
        "//mediapipe/tasks/ios/components/containers:sources/MPPEmbedding.h",
        "//mediapipe/tasks/ios/components/containers:sources/MPPEmbeddingResult.h",
        "//mediapipe/tasks/ios/components/containers:sources/MPPLandmark.h",
        "//mediapipe/tasks/ios/components/containers:sources/MPPRegionOfInterest.h",
        "//mediapipe/tasks/ios/components/processors:sources/MPPClassifierOptions.h",
        "//mediapipe/tasks/ios/core:sources/MPPBaseOptions.h",
        "//mediapipe/tasks/ios/core:sources/MPPTaskOptions.h",
        "//mediapipe/tasks/ios/core:sources/MPPTaskResult.h",
        "//mediapipe/tasks/ios/audio/audio_classifier:sources/MPPAudioClassifier.h",
        "//mediapipe/tasks/ios/audio/audio_classifier:sources/MPPAudioClassifierOptions.h",
        "//mediapipe/tasks/ios/audio/audio_classifier:sources/MPPAudioClassifierResult.h",
        "//mediapipe/tasks/ios/audio/core:sources/MPPAudioData.h",
        "//mediapipe/tasks/ios/audio/core:sources/MPPAudioDataFormat.h",
        "//mediapipe/tasks/ios/audio/core:sources/MPPAudioRecord.h",
        "//mediapipe/tasks/ios/audio/core:sources/MPPAudioRunningMode.h",
        "//mediapipe/tasks/ios/audio/core:sources/MPPFloatBuffer.h",
        "//mediapipe/tasks/ios/text/text_classifier:sources/MPPTextClassifier.h",
        "//mediapipe/tasks/ios/text/text_classifier:sources/MPPTextClassifierOptions.h",
        "//mediapipe/tasks/ios/text/text_classifier:sources/MPPTextClassifierResult.h",
        "//mediapipe/tasks/ios/text/text_embedder:sources/MPPTextEmbedder.h",
        "//mediapipe/tasks/ios/text/text_embedder:sources/MPPTextEmbedderOptions.h",
        "//mediapipe/tasks/ios/text/text_embedder:sources/MPPTextEmbedderResult.h",
        "//mediapipe/tasks/ios/text/language_detector:sources/MPPLanguageDetector.h",
        "//mediapipe/tasks/ios/text/language_detector:sources/MPPLanguageDetectorOptions.h",
        "//mediapipe/tasks/ios/text/language_detector:sources/MPPLanguageDetectorResult.h",
        "//mediapipe/tasks/ios/vision/core:sources/MPPRunningMode.h",
        "//mediapipe/tasks/ios/vision/core:sources/MPPImage.h",
        "//mediapipe/tasks/ios/vision/core:sources/MPPMask.h",
        "//mediapipe/tasks/ios/vision/face_detector:sources/MPPFaceDetector.h",
        "//mediapipe/tasks/ios/vision/face_detector:sources/MPPFaceDetectorOptions.h",
        "//mediapipe/tasks/ios/vision/face_detector:sources/MPPFaceDetectorResult.h",
        "//mediapipe/tasks/ios/vision/face_landmarker:sources/MPPFaceLandmarker.h",
        "//mediapipe/tasks/ios/vision/face_landmarker:sources/MPPFaceLandmarkerOptions.h",
        "//mediapipe/tasks/ios/vision/face_landmarker:sources/MPPFaceLandmarkerResult.h",
        "//mediapipe/tasks/ios/vision/hand_landmarker:sources/MPPHandLandmarker.h",
        "//mediapipe/tasks/ios/vision/hand_landmarker:sources/MPPHandLandmarkerOptions.h",
        "//mediapipe/tasks/ios/vision/hand_landmarker:sources/MPPHandLandmarkerResult.h",
        "//mediapipe/tasks/ios/vision/holistic_landmarker:sources/MPPHolisticLandmarker.h",
        "//mediapipe/tasks/ios/vision/holistic_landmarker:sources/MPPHolisticLandmarkerOptions.h",
        "//mediapipe/tasks/ios/vision/holistic_landmarker:sources/MPPHolisticLandmarkerResult.h",
        "//mediapipe/tasks/ios/vision/gesture_recognizer:sources/MPPGestureRecognizer.h",
        "//mediapipe/tasks/ios/vision/gesture_recognizer:sources/MPPGestureRecognizerOptions.h",
        "//mediapipe/tasks/ios/vision/gesture_recognizer:sources/MPPGestureRecognizerResult.h",
        "//mediapipe/tasks/ios/vision/image_classifier:sources/MPPImageClassifier.h",
        "//mediapipe/tasks/ios/vision/image_classifier:sources/MPPImageClassifierOptions.h",
        "//mediapipe/tasks/ios/vision/image_classifier:sources/MPPImageClassifierResult.h",
        "//mediapipe/tasks/ios/vision/image_embedder:sources/MPPImageEmbedder.h",
        "//mediapipe/tasks/ios/vision/image_embedder:sources/MPPImageEmbedderOptions.h",
        "//mediapipe/tasks/ios/vision/image_embedder:sources/MPPImageEmbedderResult.h",
        "//mediapipe/tasks/ios/vision/image_segmenter:sources/MPPImageSegmenter.h",
        "//mediapipe/tasks/ios/vision/image_segmenter:sources/MPPImageSegmenterOptions.h",
        "//mediapipe/tasks/ios/vision/image_segmenter:sources/MPPImageSegmenterResult.h",
        "//mediapipe/tasks/ios/vision/interactive_segmenter:sources/MPPInteractiveSegmenter.h",
        "//mediapipe/tasks/ios/vision/interactive_segmenter:sources/MPPInteractiveSegmenterOptions.h",
        "//mediapipe/tasks/ios/vision/interactive_segmenter:sources/MPPInteractiveSegmenterResult.h",
        "//mediapipe/tasks/ios/vision/object_detector:sources/MPPObjectDetector.h",
        "//mediapipe/tasks/ios/vision/object_detector:sources/MPPObjectDetectorOptions.h",
        "//mediapipe/tasks/ios/vision/object_detector:sources/MPPObjectDetectorResult.h",
        "//mediapipe/tasks/ios/vision/pose_landmarker:sources/MPPPoseLandmarker.h",
        "//mediapipe/tasks/ios/vision/pose_landmarker:sources/MPPPoseLandmarkerOptions.h",
        "//mediapipe/tasks/ios/vision/pose_landmarker:sources/MPPPoseLandmarkerResult.h",
    ],
)

mediapipe_static_xcframework(
    name = "MediaPipeTasksAudio_framework",
    bundle_name = "MediaPipeTasksAudio",
    public_hdrs = [
        ":MPPAudioClassifier.h",
        ":MPPAudioClassifierOptions.h",
        ":MPPAudioClassifierResult.h",
        ":MPPAudioData.h",
        ":MPPAudioDataFormat.h",
        ":MPPAudioRecord.h",
        ":MPPAudioRunningMode.h",
        ":MPPBaseOptions.h",
        ":MPPCategory.h",
        ":MPPClassificationResult.h",
        ":MPPCommon.h",
        ":MPPEmbedding.h",
        ":MPPEmbeddingResult.h",
        ":MPPFloatBuffer.h",
        ":MPPTaskOptions.h",
        ":MPPTaskResult.h",
    ],
)

mediapipe_static_xcframework(
    name = "MediaPipeTasksText_framework",
    bundle_name = "MediaPipeTasksText",
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
        ":MPPLanguageDetector.h",
        ":MPPLanguageDetectorOptions.h",
        ":MPPLanguageDetectorResult.h",
    ],
)

mediapipe_static_xcframework(
    name = "MediaPipeTasksVision_framework",
    bundle_name = "MediaPipeTasksVision",
    public_hdrs = [
        ":MPPBaseOptions.h",
        ":MPPCategory.h",
        ":MPPClassificationResult.h",
        ":MPPClassifierOptions.h",
        ":MPPDetection.h",
        ":MPPEmbedding.h",
        ":MPPEmbeddingResult.h",
        ":MPPLandmark.h",
        ":MPPRegionOfInterest.h",
        ":MPPConnection.h",
        ":MPPCommon.h",
        ":MPPTaskOptions.h",
        ":MPPTaskResult.h",
        ":MPPImage.h",
        ":MPPMask.h",
        ":MPPRunningMode.h",
        ":MPPFaceDetector.h",
        ":MPPFaceDetectorOptions.h",
        ":MPPFaceDetectorResult.h",
        ":MPPFaceLandmarker.h",
        ":MPPFaceLandmarkerOptions.h",
        ":MPPFaceLandmarkerResult.h",
        ":MPPImageClassifier.h",
        ":MPPImageClassifierOptions.h",
        ":MPPImageClassifierResult.h",
        ":MPPImageEmbedder.h",
        ":MPPImageEmbedderOptions.h",
        ":MPPImageEmbedderResult.h",
        ":MPPImageSegmenter.h",
        ":MPPImageSegmenterOptions.h",
        ":MPPImageSegmenterResult.h",
        ":MPPInteractiveSegmenter.h",
        ":MPPInteractiveSegmenterOptions.h",
        ":MPPInteractiveSegmenterResult.h",
        ":MPPHandLandmarker.h",
        ":MPPHandLandmarkerOptions.h",
        ":MPPHandLandmarkerResult.h",
        ":MPPHolisticLandmarker.h",
        ":MPPHolisticLandmarkerOptions.h",
        ":MPPHolisticLandmarkerResult.h",
        ":MPPGestureRecognizer.h",
        ":MPPGestureRecognizerOptions.h",
        ":MPPGestureRecognizerResult.h",
        ":MPPObjectDetector.h",
        ":MPPObjectDetectorOptions.h",
        ":MPPObjectDetectorResult.h",
        ":MPPPoseLandmarker.h",
        ":MPPPoseLandmarkerOptions.h",
        ":MPPPoseLandmarkerResult.h",
    ],
)

mediapipe_static_xcframework(
    name = "MediaPipeTaskGraphs_library",
    bundle_name = "MediaPipeTaskGraphs_library",
    deps = [":MediaPipeTaskGraphs"],
)

mediapipe_static_xcframework(
    name = "MediaPipeTasksCommon_framework",
    avoid_deps = CALCULATORS_AND_GRAPHS,
    bundle_name = "MediaPipeTasksCommon",
    # Including `OpenCV` and `TensorFlowLiteC` deps of the graphs here to avoid
    # conflicts with `TensorFlowLiteSwift` and `OpenCV` frameworks as iOS
    # `xcframeworks` handle duplicate dependencies with other iOS
    # `xcframeworks`. Including them with task graphs causes duplicate symbols
    # when installed alongside the respective frameworks since task graphs are
    # built as static libraries and force loaded.
    #
    # FOR SPM: Including CALCULATORS_AND_GRAPHS directly in MediaPipeTasksCommon
    # to avoid undefined symbols. Vision/Audio/Text frameworks use avoid_deps
    # to prevent duplicating these, so using multiple frameworks together is safe.
    # NOTE: conversion_data is required for ICU symbols when using -all_load.
    deps = MEDIAPIPE_TASKS_COMMON_DEPS + CALCULATORS_AND_GRAPHS + [
        "@org_tensorflow//third_party/icu/data:conversion_data",
    ] + select(OPENCV_DEPS),
)
