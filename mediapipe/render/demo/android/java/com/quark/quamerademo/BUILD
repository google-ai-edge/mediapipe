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

licenses(["notice"])

package(default_visibility = ["//visibility:private"])

cc_binary(
    name = "libola_render_jni.so",
    linkshared = 1,
    linkstatic = 1,
    srcs = [
        "//mediapipe/render/module/render_queue:olarender",
    ],
)

cc_library(
    name = "ola_render_jni_lib",
    srcs = [":libola_render_jni.so"],
    alwayslink = 1,
)

android_binary(
    name = "quamerademo",
    srcs = glob(["*.java"]),
    assets = [],
    assets_dir = "",
    manifest = "//mediapipe/render/demo/android/java/com/quark/quamerademo:AndroidManifest.xml",
    multidex = "native",
    deps = [
        ":ola_render_jni_lib",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
    ],
)
