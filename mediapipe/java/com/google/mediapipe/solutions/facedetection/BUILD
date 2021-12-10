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
    name = "facedetection",
    srcs = [
        "FaceDetection.java",
        "FaceDetectionOptions.java",
        "FaceDetectionResult.java",
        "FaceKeypoint.java",
    ],
    assets = [
        "//mediapipe/modules/face_detection:face_detection_full_range_image.binarypb",
        "//mediapipe/modules/face_detection:face_detection_full_range_sparse.tflite",
        "//mediapipe/modules/face_detection:face_detection_short_range.tflite",
        "//mediapipe/modules/face_detection:face_detection_short_range_image.binarypb",
    ],
    assets_dir = "",
    javacopts = ["-Acom.google.auto.value.AutoBuilderIsUnstable"],
    manifest = ":AndroidManifest.xml",
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/formats:detection_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/solutioncore:solution_base",
        "//third_party:autovalue",
        "@maven//:androidx_annotation_annotation",
        "@maven//:com_google_code_findbugs_jsr305",
        "@maven//:com_google_guava_guava",
    ],
)
