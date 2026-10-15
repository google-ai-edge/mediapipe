# Copyright 2026 The MediaPipe Authors.
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

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

cc_library(
    name = "text_summarizer_jni_lib",
    srcs = [
        "text_summarizer_jni.cc",
        "text_summarizer_jni.h",
    ],
    deps = [
        "//mediapipe/java/com/google/mediapipe/framework/jni:jni_util",
        "//mediapipe/tasks/c/core:base_options",
        "//mediapipe/tasks/c/core:base_options_converter",
        "//mediapipe/tasks/c/core:mp_status",
        "//mediapipe/tasks/c/text/text_summarizer:text_summarizer_lib",
    ] + select({
        "//conditions:default": [
        ],
        "//mediapipe:android": [],
    }),
    alwayslink = 1,
)
