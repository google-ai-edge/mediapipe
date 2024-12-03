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

load("//mediapipe/java/com/google/mediapipe:mediapipe_aar.bzl", "mediapipe_aar")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

android_library(
    name = "solution_info",
    srcs = ["SolutionInfo.java"],
    deps = [
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "solution_base",
    srcs = glob(
        ["*.java"],
        exclude = [
            "CameraInput.java",
            "ResultGlRenderer.java",
            "SolutionGlSurfaceView.java",
            "SolutionGlSurfaceViewRenderer.java",
            "VideoInput.java",
        ],
    ),
    visibility = ["//visibility:public"],
    deps = [
        ":logging",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/glutil",
        "//third_party:autovalue",
        "@com_google_protobuf//:protobuf_javalite",
        "@maven//:com_google_code_findbugs_jsr305",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "camera_input",
    srcs = ["CameraInput.java"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/java/com/google/mediapipe/components:android_camerax_helper",
        "//mediapipe/java/com/google/mediapipe/components:android_components",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "video_input",
    srcs = ["VideoInput.java"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/java/com/google/mediapipe/components:android_components",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/glutil",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "solution_rendering",
    srcs = [
        "ResultGlRenderer.java",
        "SolutionGlSurfaceView.java",
        "SolutionGlSurfaceViewRenderer.java",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":solution_base",
        "//mediapipe/java/com/google/mediapipe/components:android_components",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/glutil",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "logging",
    srcs = glob(
        ["logging/*.java"],
    ),
    visibility = ["//visibility:public"],
    deps = [
        ":solution_info",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

# Native dependencies of all MediaPipe solutions.
cc_binary(
    name = "libmediapipe_jni.so",
    linkshared = 1,
    linkstatic = 1,
    # TODO: Add more calculators to support other top-level solutions.
    deps = [
        "//mediapipe/java/com/google/mediapipe/framework/jni:mediapipe_framework_jni",
        "//mediapipe/modules/face_detection:face_detection_full_range_image",
        "//mediapipe/modules/face_detection:face_detection_short_range_image",
        "//mediapipe/modules/face_landmark:face_landmark_front_cpu_image",
        "//mediapipe/modules/face_landmark:face_landmark_front_gpu_image",
        "//mediapipe/modules/hand_landmark:hand_landmark_tracking_cpu_image",
        "//mediapipe/modules/hand_landmark:hand_landmark_tracking_gpu_image",
    ],
)

# Converts the .so cc_binary into a cc_library, to be consumed in an android_binary.
cc_library(
    name = "mediapipe_jni_lib",
    srcs = [":libmediapipe_jni.so"],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

mediapipe_aar(
    name = "solution_core",
    srcs = glob(["**/*.java"]),
    gen_libmediapipe = False,
)
