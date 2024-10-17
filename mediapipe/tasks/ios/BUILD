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

load("@build_bazel_rules_apple//apple:apple.bzl", "apple_static_xcframework")
load("@build_bazel_rules_apple//apple:apple_static_library.bzl", "apple_static_library")
load(
    "//mediapipe/framework/tool:ios.bzl",
    "MPP_TASK_MINIMUM_OS_VERSION",
    "strip_api_include_path_prefix",
)

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

config_setting(
    name = "enable_odml_cocoapods_build",
    define_values = {"ENABLE_ODML_COCOAPODS_BUILD": "1"},
    visibility = ["//visibility:public"],
)

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
    "//mediapipe/tasks/ios/components/containers/utils:MPPEmbeddingHelpers",
    "//mediapipe/tasks/ios/components/containers/utils:MPPEmbeddingResultHelpers",
    "//mediapipe/tasks/ios/components/utils:MPPCosineSimilarity",
    "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
]

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

CALCULATORS_AND_GRAPHS = [
    "//mediapipe/calculators/core:flow_limiter_calculator",
    "//mediapipe/tasks/cc/audio/audio_classifier:audio_classifier_graph",
    "//mediapipe/tasks/cc/audio/audio_embedder:audio_embedder_graph",
    "//mediapipe/tasks/cc/text/text_classifier:text_classifier_graph",
    "//mediapipe/tasks/cc/text/text_embedder:text_embedder_graph",
    "//mediapipe/tasks/cc/vision/face_detector:face_detector_graph",
    "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker_graph",
    "//mediapipe/tasks/cc/vision/face_stylizer:face_stylizer_graph",
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

OPENCV_DEPS = {
    "//third_party:opencv_ios_sim_arm64_source_build": ["@ios_opencv_source//:opencv_xcframework"],
    "//third_party:opencv_ios_arm64_source_build": ["@ios_opencv_source//:opencv_xcframework"],
    "//third_party:opencv_ios_x86_64_source_build": ["@ios_opencv_source//:opencv_xcframework"],
    "//third_party:opencv_ios_sim_fat_source_build": ["@ios_opencv_source//:opencv_xcframework"],
    "//conditions:default": ["@ios_opencv//:OpencvFramework"],
}

MEDIAPIPE_TASKS_COMMON_DEPS = OBJC_TASK_COMMON_DEPS + TENSORFLOW_LITE_C_DEPS + ["//mediapipe/gpu:metal_shared_resources"]

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
        "//mediapipe/tasks/ios/audio/audio_embedder:sources/MPPAudioEmbedder.h",
        "//mediapipe/tasks/ios/audio/audio_embedder:sources/MPPAudioEmbedderOptions.h",
        "//mediapipe/tasks/ios/audio/audio_embedder:sources/MPPAudioEmbedderResult.h",
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
        "//mediapipe/tasks/ios/vision/face_stylizer:sources/MPPFaceStylizer.h",
        "//mediapipe/tasks/ios/vision/face_stylizer:sources/MPPFaceStylizerOptions.h",
        "//mediapipe/tasks/ios/vision/face_stylizer:sources/MPPFaceStylizerResult.h",
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
        "//mediapipe/tasks/cc/genai/inference/c:llm_inference_engine.h",
    ],
)

apple_static_xcframework(
    name = "MediaPipeTasksAudio_framework",
    # Avoid dependencies of ":MediaPipeTasksCommon_framework" and
    # ":MediaPipeTaskGraphs_library in order to prevent duplicate symbols error
    # when the frameworks are imported in iOS projects.
    avoid_deps = MEDIAPIPE_TASKS_COMMON_DEPS + CALCULATORS_AND_GRAPHS,
    bundle_name = "MediaPipeTasksAudio",
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
        ":MPPAudioClassifier.h",
        ":MPPAudioClassifierOptions.h",
        ":MPPAudioClassifierResult.h",
        ":MPPAudioEmbedder.h",
        ":MPPAudioEmbedderOptions.h",
        ":MPPAudioEmbedderResult.h",
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
    deps = [
        "//mediapipe/tasks/ios/audio/audio_classifier:MPPAudioClassifier",
        "//mediapipe/tasks/ios/audio/audio_embedder:MPPAudioEmbedder",
    ],
)

apple_static_xcframework(
    name = "MediaPipeTasksText_framework",
    # Avoid dependencies of ":MediaPipeTasksCommon_framework" and
    # ":MediaPipeTaskGraphs_library in order to prevent duplicate symbols error
    # when the frameworks are imported in iOS projects.
    avoid_deps = MEDIAPIPE_TASKS_COMMON_DEPS + CALCULATORS_AND_GRAPHS,
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
        ":MPPLanguageDetector.h",
        ":MPPLanguageDetectorOptions.h",
        ":MPPLanguageDetectorResult.h",
    ],
    deps = [
        "//mediapipe/tasks/ios/text/language_detector:MPPLanguageDetector",
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
    avoid_deps = MEDIAPIPE_TASKS_COMMON_DEPS + CALCULATORS_AND_GRAPHS,
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
        ":MPPFaceStylizer.h",
        ":MPPFaceStylizerOptions.h",
        ":MPPFaceStylizerResult.h",
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
    deps = [
        "//mediapipe/tasks/ios/vision/face_detector:MPPFaceDetector",
        "//mediapipe/tasks/ios/vision/face_landmarker:MPPFaceLandmarker",
        "//mediapipe/tasks/ios/vision/face_stylizer:MPPFaceStylizer",
        "//mediapipe/tasks/ios/vision/gesture_recognizer:MPPGestureRecognizer",
        "//mediapipe/tasks/ios/vision/hand_landmarker:MPPHandLandmarker",
        "//mediapipe/tasks/ios/vision/holistic_landmarker:MPPHolisticLandmarker",
        "//mediapipe/tasks/ios/vision/image_classifier:MPPImageClassifier",
        "//mediapipe/tasks/ios/vision/image_embedder:MPPImageEmbedder",
        "//mediapipe/tasks/ios/vision/image_segmenter:MPPImageSegmenter",
        "//mediapipe/tasks/ios/vision/interactive_segmenter:MPPInteractiveSegmenter",
        "//mediapipe/tasks/ios/vision/object_detector:MPPObjectDetector",
        "//mediapipe/tasks/ios/vision/pose_landmarker:MPPPoseLandmarker",
    ],
)

apple_static_library(
    name = "MediaPipeTaskGraphs_library",
    # There is no way to turn off zlib dependency in custom opencv builds.
    # Hence zlib is avoided to prevent duplicate symbols because of conflicts
    # between opencv's zlib and "@zlib//:zlib"
    #
    # Avoiding `OpenCV` and `TensorFlowLiteC` deps of the graphs here to prevent
    # conflicts with `TensorFlowLiteSwift` and `OpenCV` frameworks since
    # this static library is force loaded. They are include with
    # `MediaPipeTasksCommon` xcframework instead.
    avoid_deps = TENSORFLOW_LITE_C_DEPS + [
        "@zlib//:zlib",
        "//mediapipe/gpu:metal_shared_resources",
    ] + select(OPENCV_DEPS),
    minimum_os_version = MPP_TASK_MINIMUM_OS_VERSION,
    platform_type = "ios",
    deps = CALCULATORS_AND_GRAPHS + [
        "@org_tensorflow//third_party/icu/data:conversion_data",
    ] + select(OPENCV_DEPS),
)

apple_static_xcframework(
    name = "MediaPipeTasksCommon_framework",
    # avoids gpu targets since they will be built with
    # ":MediaPipeTaskGraphs_library". Otherwise it will result in
    # duplicate symbols error when the frameworks are imported in iOS.
    avoid_deps = [
        "@zlib//:zlib",
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
    # Including `OpenCV` and `TensorFlowLiteC` deps of the graphs here to avoid
    # conflicts with `TensorFlowLiteSwift` and `OpenCV` frameworks as iOS
    # `xcframeworks` handle duplicate dependencies with other iOS
    # `xcframeworks`. Including them with task graphs causes duplicate symbols
    # when installed alongside the respective frameworks since task graphs are
    # built as static libraries and force loaded.
    deps = MEDIAPIPE_TASKS_COMMON_DEPS + select(OPENCV_DEPS),
)

apple_static_xcframework(
    name = "MediaPipeTasksGenAI_framework",
    avoid_deps = [
        "//mediapipe/tasks/cc/genai/inference/c:libllm_inference_engine_cpu",
    ],
    bundle_name = "MediaPipeTasksGenAI",
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
    deps = ["//mediapipe/tasks/ios/genai/inference:LlmInference"],
)

apple_static_xcframework(
    name = "MediaPipeTasksGenAIC_framework",
    bundle_name = "MediaPipeTasksGenAIC",
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
        ":llm_inference_engine.h",
    ],
    # Including here the deps that were in avoid_deps in MediaPipeTasksGenAI_library.
    deps = TENSORFLOW_LITE_C_DEPS + select({
        "//conditions:default": ["//mediapipe/tasks/cc/genai/inference/c:libllm_inference_engine_cpu"],
        ":enable_odml_cocoapods_build": ["@odml//odml/infra/genai/inference/c:libllm_inference_engine"],
    }),
)

apple_static_library(
    name = "MediaPipeTasksGenAI_library",
    # Excluding these to avoid duplicate symbols.
    avoid_deps = TENSORFLOW_LITE_C_DEPS + [
        "//mediapipe/gpu:metal_shared_resources",
    ],
    minimum_os_version = MPP_TASK_MINIMUM_OS_VERSION,
    platform_type = "ios",
    deps = select({
        "//conditions:default": [],
        ":enable_odml_cocoapods_build": ["@odml//odml/infra/genai/inference:all_possible_calculators"],
    }),
)
