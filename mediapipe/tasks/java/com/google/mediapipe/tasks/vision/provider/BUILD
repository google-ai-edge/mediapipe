# Copyright 2025 The MediaPipe Authors.
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

load("@rules_kotlin//kotlin:android.bzl", "kt_android_library")
load("@rules_kotlin//kotlin:jvm.bzl", "kt_jvm_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

kt_jvm_library(
    name = "quantization",
    srcs = ["Quantization.kt"],
)

kt_jvm_library(
    name = "vision_model",
    srcs = [
        "VisionModel.kt",
        "VisionTask.kt",
        "VisionTaskImpl.kt",
    ],
    deps = [
        ":quantization",
    ],
)

kt_android_library(
    name = "provider",
    srcs = glob(["*.kt"]),
    deps = [
        "//java/com/google/android/apps/play/store/sdk/aipacks:1p_client_ai_delivery",
        "//java/com/google/android/apps/play/store/sdk/splitinstall:1p_client_split_install",
        "//mediapipe/java/com/google/mediapipe/framework/image",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/processors:classifieroptions",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:facedetector",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:facelandmarker",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:gesturerecognizer",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:handlandmarker",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:imageclassifier",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:imageembedder",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:imagesegmenter",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:interactivesegmenter",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:objectdetector",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:poselandmarker",
        "//third_party/java/androidx/activity",
        "@maven//:androidx_annotation_annotation",
    ],
)
