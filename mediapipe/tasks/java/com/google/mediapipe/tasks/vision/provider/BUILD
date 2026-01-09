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
        "//mediapipe/java/com/google/mediapipe/framework/image",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:imagesegmenter",
    ],
)
