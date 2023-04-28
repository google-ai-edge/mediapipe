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

package(default_visibility = ["//visibility:public"])

android_library(
    name = "core",
    srcs = glob(["*.java"]),
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    manifest = "AndroidManifest.xml",
    deps = [
        ":logging",
        "//mediapipe/calculators/core:flow_limiter_calculator_java_proto_lite",
        "//mediapipe/calculators/tensor:inference_calculator_java_proto_lite",
        "//mediapipe/framework:calculator_java_proto_lite",
        "//mediapipe/framework:calculator_options_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/tasks/cc/core/proto:acceleration_java_proto_lite",
        "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
        "//mediapipe/tasks/cc/core/proto:external_file_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core/jni:model_resources_cache_jni",
        "//third_party:autovalue",
        "@com_google_protobuf//:protobuf_javalite",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "logging",
    srcs = glob(
        ["logging/*.java"],
    ),
    deps = [
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

load("//mediapipe/tasks/java/com/google/mediapipe/tasks:mediapipe_tasks_aar.bzl", "mediapipe_tasks_core_aar")

mediapipe_tasks_core_aar(
    name = "tasks_core",
    srcs = glob(["**/*.java"]) + [
        "//mediapipe/java/com/google/mediapipe/framework/image:java_src",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:java_src",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/processors:java_src",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/utils:java_src",
    ],
    manifest = "AndroidManifest.xml",
)
