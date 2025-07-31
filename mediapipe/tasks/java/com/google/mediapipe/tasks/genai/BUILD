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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//mediapipe/tasks/java/com/google/mediapipe/tasks:mediapipe_tasks_aar.bzl", "mediapipe_build_aar_with_jni", "mediapipe_java_proto_src_extractor", "mediapipe_jni_binary")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_jni_binary(
    name = "llm_inference_engine_jni",
    uses_explicit_exports = True,
    deps = ["//mediapipe/tasks/java/com/google/mediapipe/tasks/genai/llminference/jni:llm"],
)

cc_library(
    name = "libllm_inference_engine_jni_lib",
    srcs = [":libllm_inference_engine_jni.so"],
    alwayslink = 1,
)

android_library(
    name = "tasks_genai_aar_library",
    srcs = glob(["**/*.java"]) + [
        mediapipe_java_proto_src_extractor(
            src_out = "com/google/mediapipe/tasks/genai/llminference/jni/proto/LlmOptionsProto.java",
            target = "//mediapipe/tasks/java/com/google/mediapipe/tasks/genai/llminference/jni/proto:llm_options_java_proto_lite",
        ),
        mediapipe_java_proto_src_extractor(
            src_out = "com/google/mediapipe/tasks/genai/llminference/jni/proto/LlmResponseContextProto.java",
            target = "//mediapipe/tasks/java/com/google/mediapipe/tasks/genai/llminference/jni/proto:llm_response_context_java_proto_lite",
        ),
    ],
    exports_manifest = 1,
    manifest = "AndroidManifest.xml",
    deps = [
        ":libllm_inference_engine_jni_lib",
        "//mediapipe/java/com/google/mediapipe/framework/image",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/genai/llminference/jni/proto:llm_options_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/genai/llminference/jni/proto:llm_response_context_java_proto_lite",
        "//third_party:autovalue",
        "@com_google_protobuf//:protobuf_javalite",
        "@maven//:com_google_guava_guava",
    ],
)

mediapipe_build_aar_with_jni(
    name = "tasks_genai",
    android_library = ":tasks_genai_aar_library",
)
