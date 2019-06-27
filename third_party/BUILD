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
#

licenses(["notice"])  # Apache License 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "opencv",
    visibility = ["//visibility:public"],
    deps = select({
        "//mediapipe:android_x86": [
            "@android_opencv//:libopencv_x86",
        ],
        "//mediapipe:android_x86_64": [
            "@android_opencv//:libopencv_x86_64",
        ],
        "//mediapipe:android_armeabi": [
            "@android_opencv//:libopencv_armeabi-v7a",
        ],
        "//mediapipe:android_arm": [
            "@android_opencv//:libopencv_armeabi-v7a",
        ],
        "//mediapipe:android_arm64": [
            "@android_opencv//:libopencv_arm64-v8a",
        ],
        "//mediapipe:macos": [
            "@macos_opencv//:opencv",
        ],
        "//conditions:default": [
            "@linux_opencv//:opencv",
        ],
    }),
)

android_library(
    name = "android_constraint_layout",
    exports = [
        "@maven//:com_android_support_constraint_constraint_layout",
    ],
)

android_library(
    name = "androidx_appcompat",
    exports = [
        "@maven//:androidx_appcompat_appcompat",
    ],
)

# TODO: Get the AARs from Google's Maven Repository.
aar_import(
    name = "camerax_core",
    aar = "camera-core-1.0.0-alpha01.aar",
)

aar_import(
    name = "camera2",
    aar = "camera-camera2-1.0.0-alpha01.aar",
)
