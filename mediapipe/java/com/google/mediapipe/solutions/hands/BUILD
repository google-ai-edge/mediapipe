# Copyright 2021 The MediaPipe Authors.
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

licenses(["notice"])

android_library(
    name = "hands",
    srcs = [
        "HandLandmark.java",
        "Hands.java",
        "HandsOptions.java",
        "HandsResult.java",
    ],
    assets = [
        "//mediapipe/modules/hand_landmark:hand_landmark_full.tflite",
        "//mediapipe/modules/hand_landmark:hand_landmark_lite.tflite",
        "//mediapipe/modules/hand_landmark:hand_landmark_tracking_cpu_image.binarypb",
        "//mediapipe/modules/hand_landmark:hand_landmark_tracking_gpu_image.binarypb",
        "//mediapipe/modules/hand_landmark:handedness.txt",
        "//mediapipe/modules/palm_detection:palm_detection_full.tflite",
        "//mediapipe/modules/palm_detection:palm_detection_lite.tflite",
    ],
    assets_dir = "",
    javacopts = ["-Acom.google.auto.value.AutoBuilderIsUnstable"],
    manifest = ":AndroidManifest.xml",
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/formats:classification_java_proto_lite",
        "//mediapipe/framework/formats:landmark_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/solutioncore:solution_base",
        "//third_party:autovalue",
        "@maven//:androidx_annotation_annotation",
        "@maven//:com_google_code_findbugs_jsr305",
        "@maven//:com_google_guava_guava",
    ],
)
