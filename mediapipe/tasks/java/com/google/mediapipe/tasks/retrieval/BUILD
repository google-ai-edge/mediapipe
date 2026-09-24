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

load("//mediapipe/tasks/java/com/google/mediapipe/tasks:mediapipe_tasks_aar.bzl", "mediapipe_build_aar_with_jni", "mediapipe_jni_binary")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_jni_binary(
    name = "sqlite_vector_store_jni",
    uses_explicit_exports = True,
    deps = [
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/retrieval/jni:retrieval_jni_impl",
    ],
)

android_library(
    name = "retrieval",
    srcs = glob([
        "model/*.java",
        "components/*.java",
        "chunking/*.java",
        "semanticretriever/*.java",
        "universalembedder/*.java",
    ]),
    manifest = "AndroidManifest.xml",
    deps = [
        ":libsqlite_vector_store_jni_lib",
        "//mediapipe/calculators/tensor:inference_calculator_java_proto_lite",
        "//mediapipe/framework:calculator_options_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework_no_mff",
        "//mediapipe/java/com/google/mediapipe/framework/image",
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_java_proto_lite",
        "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
        "//mediapipe/tasks/cc/retrieval/semantic_retriever/proto:memory_java_proto_lite",
        "//mediapipe/tasks/cc/retrieval/semantic_retriever/proto:vector_stores_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:audiodata",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:embedding",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:embeddingresult",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/utils:cosinesimilarity",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core:core_java",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core:logging",
        "//third_party:autovalue",
        "@com_google_protobuf//java/lite",
        "@litert_lm//kotlin/java/com/google/ai/edge/litertlm:litertlm-android",
        "@maven//:androidx_annotation_annotation",
        "@maven//:androidx_appsearch_appsearch",
        "@maven//:androidx_appsearch_appsearch_local_storage",
        "@maven//:com_google_guava_guava",
    ],
)

mediapipe_build_aar_with_jni(
    name = "tasks_retrieval",
    android_library = ":retrieval",
)
