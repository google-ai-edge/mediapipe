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

licenses(["notice"])  # Apache 2.0

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
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/glutil",
        "//third_party:androidx_appcompat",
        "//third_party:androidx_core",
        "//third_party:androidx_legacy_support_v4",
        "//third_party:androidx_recyclerview",
        "@com_google_code_findbugs//jar",
        "@com_google_guava_android//jar",
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
        "//third_party:androidx_appcompat",
        "//third_party:androidx_legacy_support_v4",
        "//third_party:camera2",
        "//third_party:camerax_core",
        "@androidx_concurrent_futures//jar",
        "@androidx_lifecycle//jar",
        "@com_google_code_findbugs//jar",
        "@com_google_guava_android//jar",
    ],
)

android_library(
    name = "android_microphone_helper",
    srcs = [
        "MicrophoneHelper.java",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_code_findbugs//jar",
        "@com_google_guava_android//jar",
    ],
)
