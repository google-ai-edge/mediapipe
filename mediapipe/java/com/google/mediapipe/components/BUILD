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

licenses(["notice"])

android_library(
    name = "android_components",
    srcs = glob(
        ["*.java"],
        exclude = [
            "CameraHelper.java",
            "CameraXPreviewHelper.java",
        ],
    ),
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_java_proto_lite",
        "//mediapipe/framework:calculator_options_java_proto_lite",
        "//mediapipe/framework/tool:calculator_graph_template_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/glutil",
        "@maven//:androidx_appcompat_appcompat",
        "@maven//:androidx_core_core",
        "@maven//:androidx_legacy_legacy_support_v4",
        "@maven//:androidx_recyclerview_recyclerview",
        "@maven//:com_google_code_findbugs_jsr305",
        "@maven//:com_google_guava_guava",
    ],
)

# Note: We need to separate the camera helper files in a different BUILD target because CameraX has a minimum Android API
# requirement of API 21. Users of android_components may have different API dependencies.
android_library(
    name = "android_camerax_helper",
    srcs = [
        "CameraHelper.java",
        "CameraXPreviewHelper.java",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/java/com/google/mediapipe/glutil",
        "@maven//:androidx_appcompat_appcompat",
        "@maven//:androidx_camera_camera_camera2",
        "@maven//:androidx_camera_camera_core",
        "@maven//:androidx_camera_camera_lifecycle",
        "@maven//:androidx_concurrent_concurrent_futures",
        "@maven//:androidx_core_core",
        "@maven//:androidx_legacy_legacy_support_v4",
        "@maven//:androidx_lifecycle_lifecycle_common",
        "@maven//:com_google_code_findbugs_jsr305",
        "@maven//:com_google_guava_guava",
        "@maven//:com_google_guava_listenablefuture",
    ],
)

# Interfaces for common Audio Consumer and Producers in MediaPipe.
android_library(
    name = "android_audio_components",
    srcs = [
        "AudioDataConsumer.java",
        "AudioDataProcessor.java",
        "AudioDataProducer.java",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@maven//:com_google_code_findbugs_jsr305",
        "@maven//:com_google_guava_guava",
    ],
)

# MicrophoneHelper that provides access to audio data from a microphone
android_library(
    name = "android_microphone_helper",
    srcs = [
        "MicrophoneHelper.java",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":android_audio_components",
        "@maven//:com_google_guava_guava",
    ],
)

# Expose the java source files for building mediapipe AAR.
filegroup(
    name = "java_src",
    srcs = glob(["*.java"]),
    visibility = ["//mediapipe:__subpackages__"],
)
