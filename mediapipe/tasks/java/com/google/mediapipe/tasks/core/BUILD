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

load("//mediapipe/tasks/java/com/google/mediapipe/tasks:mediapipe_tasks_aar.bzl", "mediapipe_tasks_core_aar")

package(default_visibility = ["//visibility:public"])

android_library(
    name = "core",
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    manifest = "AndroidManifest.xml",
    exports = [
        ":core_java",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core/jni:model_resources_cache_jni",
    ],
    deps = ["@maven//:com_google_guava_guava"],
)

android_library(
    name = "core_java",
    srcs = glob(
        ["*.java"],
        exclude = ["LlmTaskRunner.java"],
    ),
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
        "//mediapipe/java/com/google/mediapipe/framework:android_framework_no_mff",
        "//mediapipe/tasks/cc/core/proto:acceleration_java_proto_lite",
        "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
        "//mediapipe/tasks/cc/core/proto:external_file_java_proto_lite",
        "//third_party:any_java_proto",
        "//third_party:autovalue",
        "@com_google_protobuf//:protobuf_javalite",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "llm",
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    manifest = "AndroidManifest.xml",
    exports = [
        ":llm_java",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core/jni:llm",
    ],
    deps = ["@maven//:com_google_guava_guava"],
)

filegroup(
    name = "llm_java_srcs",
    srcs = ["LlmTaskRunner.java"],
)

android_library(
    name = "llm_java",
    srcs = [":llm_java_srcs"],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    manifest = "AndroidManifest.xml",
    deps = [
        ":core_java",
        ":logging",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core/jni/proto:llm_options_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core/jni/proto:llm_response_context_java_proto_lite",
        "//third_party/java/protobuf:protobuf_lite",
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

mediapipe_tasks_core_aar(
    name = "tasks_core",
    srcs = glob(
        ["**/*.java"],
        exclude = ["LlmTaskRunner.java"],
    ) + [
        "//mediapipe/java/com/google/mediapipe/framework/image:java_src",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:java_src",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/processors:java_src",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/utils:java_src",
    ],
    manifest = "AndroidManifest.xml",
)
