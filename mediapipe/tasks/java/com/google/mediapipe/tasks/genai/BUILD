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

load("//mediapipe/tasks/java/com/google/mediapipe/tasks:mediapipe_tasks_aar.bzl", "mediapipe_tasks_genai_aar")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "libllm_inference_engine_jni.so",
    linkopts = [
        "-Wl,-soname=libllm_inference_engine_jni.so",
        "-fvisibility=hidden",
    ],
    linkshared = 1,
    deps = ["//mediapipe/tasks/java/com/google/mediapipe/tasks/core/jni:llm"],
)

cc_library(
    name = "libllm_inference_engine_jni_lib",
    srcs = [":libllm_inference_engine_jni.so"],
    alwayslink = 1,
)

mediapipe_tasks_genai_aar(
    name = "tasks_genai",
    srcs = glob(["**/*.java"]) + ["//mediapipe/tasks/java/com/google/mediapipe/tasks/core:llm_java_srcs"],
    native_library = ":libllm_inference_engine_jni_lib",
)
