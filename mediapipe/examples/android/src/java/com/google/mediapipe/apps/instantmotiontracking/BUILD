# Copyright 2020 The MediaPipe Authors.
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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
# Placeholder: load java_lite_proto_library

licenses(["notice"])

package(default_visibility = ["//visibility:private"])

java_lite_proto_library(
    name = "sticker_buffer_java_proto_lite",
    deps = ["//mediapipe/graphs/instant_motion_tracking/calculators:sticker_buffer_proto"],
)

android_library(
    name = "instantmotiontracking_lib",
    srcs = glob(["*.java"]),
    manifest = "//mediapipe/examples/android/src/java/com/google/mediapipe/apps/basic:AndroidManifest.xml",
    resource_files = glob([
        "res/layout/**",
        "res/drawable/**",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":sticker_buffer_java_proto_lite",
        "//mediapipe/examples/android/src/java/com/google/mediapipe/apps/basic:basic_lib",
        "//mediapipe/java/com/google/mediapipe/components:android_components",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//third_party:opencv",
        "@maven//:androidx_appcompat_appcompat",
        "@maven//:androidx_concurrent_concurrent_futures",
        "@maven//:androidx_core_core",
        "@maven//:com_github_bumptech_glide_glide",
        "@maven//:com_google_guava_guava",
    ],
)

# Include all calculators specific to this project defined by BUILD in graphs
cc_binary(
    name = "libmediapipe_jni.so",
    linkshared = 1,
    linkstatic = 1,
    deps = [
        "//mediapipe/graphs/instant_motion_tracking:instant_motion_tracking_deps",
        "//mediapipe/java/com/google/mediapipe/framework/jni:mediapipe_framework_jni",
    ],
)

# Converts the .so cc_binary into a cc_library, to be consumed in an android_binary.
cc_library(
    name = "mediapipe_jni_lib",
    srcs = [":libmediapipe_jni.so"],
    alwayslink = 1,
)

genrule(
    name = "asset3d",
    srcs = ["//mediapipe/examples/android/src/java/com/google/mediapipe/apps/instantmotiontracking/assets:robot/robot.obj.uuu.zip"],
    outs = ["robot/robot.obj.uuu"],
    cmd = "unzip -p $< > $@",
)

android_binary(
    name = "instantmotiontracking",
    assets = [
        ":asset3d",
        "//mediapipe/graphs/instant_motion_tracking:instant_motion_tracking.binarypb",
        "//mediapipe/examples/android/src/java/com/google/mediapipe/apps/instantmotiontracking/assets:gif/gif.obj.uuu",
        "//mediapipe/examples/android/src/java/com/google/mediapipe/apps/instantmotiontracking/assets:gif/default_gif_texture.jpg",
        "//mediapipe/examples/android/src/java/com/google/mediapipe/apps/instantmotiontracking/assets:robot/robot_texture.jpg",
    ],
    assets_dir = "",
    manifest = "//mediapipe/examples/android/src/java/com/google/mediapipe/apps/basic:AndroidManifest.xml",
    manifest_values = {
        "applicationId": "com.google.mediapipe.apps.instantmotiontracking",
        "appName": "Instant Motion Tracking",
        "mainActivity": ".MainActivity",
        "cameraFacingFront": "False",
        "binaryGraphName": "instant_motion_tracking.binarypb",
        "inputVideoStreamName": "input_video",
        "outputVideoStreamName": "output_video",
        "flipFramesVertically": "True",
        "converterNumBuffers": "2",
    },
    multidex = "native",
    deps = [
        ":instantmotiontracking_lib",
        ":mediapipe_jni_lib",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//third_party:libc++_shared_lib",
    ],
)
