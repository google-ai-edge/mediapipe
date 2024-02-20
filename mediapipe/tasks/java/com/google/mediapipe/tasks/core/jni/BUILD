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

load("@org_tensorflow//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(default_visibility = [
    "//mediapipe/tasks:internal",
    "//third_party/generative_computing:__subpackages__",
])

licenses(["notice"])

cc_library_with_tflite(
    name = "model_resources_cache_jni",
    srcs = [
        "model_resources_cache_jni.cc",
    ],
    hdrs = [
        "model_resources_cache_jni.h",
    ] + select({
        "//conditions:default": [
            ":jni.h",
            ":jni_md.h",
        ],
        # The Android toolchain makes "jni.h" available in the include path.
        # For non-Android toolchains, generate jni.h and jni_md.h.
        "//mediapipe:android": [],
    }),
    tflite_deps = [
        "//mediapipe/tasks/cc/core:model_resources_cache",
    ],
    deps = [
        "//mediapipe/java/com/google/mediapipe/framework/jni:mediapipe_framework_jni",
        "//mediapipe/tasks/cc/core:mediapipe_builtin_op_resolver",
    ] + select({
        "//conditions:default": [
        ],
        "//mediapipe:android": [],
    }),
    alwayslink = 1,
)

cc_library(
    name = "llm_inference_engine_hdr",
    hdrs = ["llm_inference_engine.h"],
)

cc_library(
    name = "llm",
    srcs = ["llm.cc"],
    hdrs = ["llm.h"],
    deps = [
        ":llm_inference_engine_hdr",
        "//mediapipe/java/com/google/mediapipe/framework/jni:class_registry",
        "//mediapipe/java/com/google/mediapipe/framework/jni:jni_util",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core/jni/proto:llm_options_cc_proto",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core/jni/proto:llm_response_context_cc_proto",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@odml//odml/infra/genai/inference/c:libllm_inference_engine",
        "@odml//odml/infra/genai/inference/c:libllm_inference_engine_deps",
    ] + select({
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
