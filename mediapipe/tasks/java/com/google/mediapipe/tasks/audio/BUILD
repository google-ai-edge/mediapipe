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
load("//mediapipe/tasks/java/com/google/mediapipe/tasks:mediapipe_tasks_aar.bzl", "mediapipe_build_aar_with_jni", "mediapipe_jni_binary")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

android_library(
    name = "core",
    srcs = glob(["core/*.java"]),
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    deps = [
        ":libmediapipe_tasks_audio_jni_lib",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework_no_mff",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:audiodata",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
        "@maven//:com_google_guava_guava",
    ],
)

# The native library of all MediaPipe audio tasks.
mediapipe_jni_binary(
    name = "mediapipe_tasks_audio_jni",
    deps = [
        "//mediapipe/java/com/google/mediapipe/framework/jni:mediapipe_framework_jni",
        "//mediapipe/tasks/cc/audio/audio_classifier:audio_classifier_graph",
        "//mediapipe/tasks/cc/audio/audio_embedder:audio_embedder_graph",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core/jni:model_resources_cache_jni",
    ],
)

cc_library(
    name = "libmediapipe_tasks_audio_jni_lib",
    srcs = [":libmediapipe_tasks_audio_jni.so"],
    alwayslink = 1,
)

android_library(
    name = "audioclassifier",
    srcs = [
        "audioclassifier/AudioClassifier.java",
        "audioclassifier/AudioClassifierResult.java",
    ],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    manifest = "audioclassifier/AndroidManifest.xml",
    deps = [
        ":core",
        "//mediapipe/framework:calculator_options_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/tasks/cc/audio/audio_classifier/proto:audio_classifier_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_java_proto_lite",
        "//mediapipe/tasks/cc/components/processors/proto:classifier_options_java_proto_lite",
        "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:audiodata",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:classificationresult",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "audioembedder",
    srcs = [
        "audioembedder/AudioEmbedder.java",
        "audioembedder/AudioEmbedderResult.java",
    ],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    manifest = "audioembedder/AndroidManifest.xml",
    deps = [
        ":core",
        "//mediapipe/framework:calculator_options_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/tasks/cc/audio/audio_embedder/proto:audio_embedder_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_java_proto_lite",
        "//mediapipe/tasks/cc/components/processors/proto:embedder_options_java_proto_lite",
        "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/audio:libmediapipe_tasks_audio_jni_lib",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:audiodata",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:embeddingresult",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "tasks_audio_aar_library",
    srcs = glob(["**/*.java"]),
    manifest = "AndroidManifest.xml",
    deps = [
        ":libmediapipe_tasks_audio_jni_lib",
        "//mediapipe/framework:calculator_options_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework_no_mff",
        "//mediapipe/tasks/cc/audio/audio_classifier/proto:audio_classifier_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/audio/audio_embedder/proto:audio_embedder_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_java_proto_lite",
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_java_proto_lite",
        "//mediapipe/tasks/cc/components/processors/proto:classifier_options_java_proto_lite",
        "//mediapipe/tasks/cc/components/processors/proto:embedder_options_java_proto_lite",
        "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:audiodata",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:classificationresult",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:embeddingresult",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core:core_java",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

mediapipe_build_aar_with_jni(
    name = "tasks_audio",
    android_library = ":tasks_audio_aar_library",
)
