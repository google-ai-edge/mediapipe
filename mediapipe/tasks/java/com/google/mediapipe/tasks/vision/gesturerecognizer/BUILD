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
    name = "gesturerecognizer",
    srcs = [
        "GestureRecognitionResult.java",
    ],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    manifest = ":AndroidManifest.xml",
    deps = [
        "//mediapipe/framework:calculator_options_java_proto_lite",
        "//mediapipe/framework/formats:classification_java_proto_lite",
        "//mediapipe/framework/formats:landmark_java_proto_lite",
        "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/gesture_recognizer/proto:gesture_recognizer_graph_options_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:category",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:landmark",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)
