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

load("//mediapipe/tasks/java/com/google/mediapipe/tasks:mediapipe_tasks_aar.bzl", "mediapipe_tasks_text_aar")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

# The native library of all MediaPipe text tasks.
cc_binary(
    name = "libmediapipe_tasks_text_jni.so",
    linkopts = [
        "-Wl,--no-undefined",
        "-Wl,--version-script,$(location //mediapipe/tasks/java:version_script.lds)",
    ],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        "//mediapipe/java/com/google/mediapipe/framework/jni:mediapipe_framework_jni",
        "//mediapipe/tasks/cc/text/text_classifier:text_classifier_graph",
        "//mediapipe/tasks/cc/text/text_embedder:text_embedder_graph",
        "//mediapipe/tasks/java:version_script.lds",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core/jni:model_resources_cache_jni",
    ],
)

cc_library(
    name = "libmediapipe_tasks_text_jni_lib",
    srcs = [":libmediapipe_tasks_text_jni.so"],
    alwayslink = 1,
)

android_library(
    name = "textclassifier",
    srcs = [
        "textclassifier/TextClassifier.java",
        "textclassifier/TextClassifierResult.java",
    ],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    manifest = "textclassifier/AndroidManifest.xml",
    deps = [
        "//mediapipe/framework:calculator_options_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_java_proto_lite",
        "//mediapipe/tasks/cc/components/processors/proto:classifier_options_java_proto_lite",
        "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
        "//mediapipe/tasks/cc/text/text_classifier/proto:text_classifier_graph_options_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:classificationresult",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/text:libmediapipe_tasks_text_jni_lib",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "textembedder",
    srcs = [
        "textembedder/TextEmbedder.java",
        "textembedder/TextEmbedderResult.java",
    ],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    manifest = "textembedder/AndroidManifest.xml",
    deps = [
        "//mediapipe/framework:calculator_options_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_java_proto_lite",
        "//mediapipe/tasks/cc/components/processors/proto:embedder_options_java_proto_lite",
        "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
        "//mediapipe/tasks/cc/text/text_embedder/proto:text_embedder_graph_options_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:embedding",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:embeddingresult",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/utils:cosinesimilarity",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/text:libmediapipe_tasks_text_jni_lib",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "languagedetector",
    srcs = [
        "languagedetector/LanguageDetector.java",
        "languagedetector/LanguageDetectorResult.java",
        "languagedetector/LanguagePrediction.java",
    ],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    manifest = "textembedder/AndroidManifest.xml",
    deps = [
        ":libmediapipe_tasks_text_jni_lib",
        "//mediapipe/framework:calculator_options_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_java_proto_lite",
        "//mediapipe/tasks/cc/components/processors/proto:classifier_options_java_proto_lite",
        "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
        "//mediapipe/tasks/cc/text/text_classifier/proto:text_classifier_graph_options_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:category",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:classificationresult",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:classifications",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

mediapipe_tasks_text_aar(
    name = "tasks_text",
    srcs = glob(["**/*.java"]),
    native_library = ":libmediapipe_tasks_text_jni_lib",
)
