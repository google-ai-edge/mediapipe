# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

android_library(
    name = "audiodata",
    srcs = ["AudioData.java"],
    deps = [
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "category",
    srcs = ["Category.java"],
    deps = [
        "//mediapipe/framework/formats:classification_java_proto_lite",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "detection",
    srcs = ["Detection.java"],
    deps = [
        ":category",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "classifications",
    srcs = ["Classifications.java"],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    deps = [
        ":category",
        "//mediapipe/framework/formats:classification_java_proto_lite",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_java_proto_lite",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "classificationresult",
    srcs = ["ClassificationResult.java"],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    deps = [
        ":classifications",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_java_proto_lite",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "handlandmark",
    srcs = ["HandLandmark.java"],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    deps = [
        "@maven//:androidx_annotation_annotation",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "landmark",
    srcs = ["Landmark.java"],
    deps = [
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

# Expose the java source files for building mediapipe tasks core AAR.
filegroup(
    name = "java_src",
    srcs = glob(["*.java"]),
    visibility = ["//mediapipe/tasks/java/com/google/mediapipe/tasks/core:__subpackages__"],
)

android_library(
    name = "embedding",
    srcs = ["Embedding.java"],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    deps = [
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_java_proto_lite",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "embeddingresult",
    srcs = ["EmbeddingResult.java"],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    deps = [
        ":embedding",
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_java_proto_lite",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)
