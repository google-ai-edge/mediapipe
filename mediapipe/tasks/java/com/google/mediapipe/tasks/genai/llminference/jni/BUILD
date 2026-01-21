# Copyright 2022 The MediaPipe Authors.
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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = [
    "//javatests/com/google/android/apps/inputmethod:__subpackages__",
    "//mediapipe/tasks:internal",
])

licenses(["notice"])

cc_library(
    name = "llm",
    srcs = ["llm.cc"],
    hdrs = ["llm.h"],
    deps = [
        "//mediapipe/java/com/google/mediapipe/framework/jni:class_registry",
        "//mediapipe/java/com/google/mediapipe/framework/jni:jni_util",
        "//mediapipe/tasks/cc/genai/inference/c:libllm_inference_engine_cpu",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/genai/llminference/jni/proto:llm_options_cc_proto",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/genai/llminference/jni/proto:llm_response_context_cc_proto",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@skia//:core",
    ] + select({
        "//conditions:default": [
        ],
        "//mediapipe:android": [],
    }),
    alwayslink = 1,
)

cc_binary(
    name = "llm_jni",
    linkshared = 1,
    deps = [":llm"],
)

genrule(
    name = "copy_jni_h",
    srcs = ["@bazel_tools//tools/jdk:jni_header"],
    outs = ["jni.h"],
    cmd = "cp -f $< $@",
)

genrule(
    name = "copy_jni_md_h",
    srcs = select({
        "//mediapipe:macos": ["@bazel_tools//tools/jdk:jni_md_header-darwin"],
        "//conditions:default": ["@bazel_tools//tools/jdk:jni_md_header-linux"],
    }),
    outs = ["jni_md.h"],
    cmd = "cp -f $< $@",
)
