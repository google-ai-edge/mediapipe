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

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

cc_library_with_tflite(
    name = "model_resources_cache_jni",
    srcs = [
        "model_resources_cache_jni.cc",
    ],
    hdrs = [
        "model_resources_cache_jni.h",
    ],
    tflite_deps = [
        "//mediapipe/tasks/cc/core:model_resources_cache",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
    deps = [
        "//mediapipe/java/com/google/mediapipe/framework/jni:mediapipe_framework_jni",
        "//mediapipe/tasks/cc/core:mediapipe_builtin_op_resolver",
    ] + select({
        "//conditions:default": ["//third_party/java/jdk:jni"],
        "//mediapipe:android": [],
    }),
    alwayslink = 1,
)
