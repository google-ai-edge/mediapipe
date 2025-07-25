# Copyright 2023 The MediaPipe Authors.
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
load(
    "//mediapipe/framework/tool:mediapipe_files.bzl",
    "mediapipe_files",
)
load("//mediapipe/tasks/java/com/google/mediapipe/tasks:mediapipe_tasks_aar.bzl", "mediapipe_build_aar_with_jni")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_files(srcs = ["libimagegenerator_gpu.so"])

# The native library of MediaPipe vision image generator tasks.
cc_binary(
    name = "libmediapipe_tasks_vision_image_generator_jni.so",
    linkopts = [
        "-Wl,--no-undefined",
        "-Wl,--version-script,$(location //mediapipe/tasks/java:version_script.lds)",
    ],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/java/com/google/mediapipe/framework/jni:mediapipe_framework_jni",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker_graph",
        "//mediapipe/tasks/cc/vision/image_generator:image_generator_graph",
        "//mediapipe/tasks/cc/vision/image_segmenter:image_segmenter_graph",
        "//mediapipe/tasks/java:version_script.lds",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core/jni:model_resources_cache_jni",
    ],
)

cc_library(
    name = "libmediapipe_tasks_vision_image_generator_jni_lib",
    srcs = [
        ":libimagegenerator_gpu.so",
        ":libmediapipe_tasks_vision_image_generator_jni.so",
    ],
    alwayslink = 1,
)

android_library(
    name = "imagegenerator",
    srcs = [
        "ImageGenerator.java",
        "ImageGeneratorResult.java",
    ],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    manifest = "AndroidManifest.xml",
    deps = [
        "//mediapipe/framework:calculator_options_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/framework/image",
        "//mediapipe/tasks/cc/core/proto:external_file_java_proto_lite",
        "//mediapipe/tasks/cc/vision/face_landmarker/proto:face_landmarker_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/image_generator/diffuser:stable_diffusion_iterate_calculator_java_proto_lite",
        "//mediapipe/tasks/cc/vision/image_generator/proto:conditioned_image_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/image_generator/proto:control_plugin_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/image_generator/proto:image_generator_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/image_segmenter/proto:image_segmenter_graph_options_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:core_java",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:facelandmarker",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:imagesegmenter",
        "//third_party:any_java_proto",
        "//third_party:autovalue",
        "@com_google_protobuf//:protobuf_javalite",
        "@maven//:androidx_annotation_annotation",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "tasks_vision_image_generator_aar_library",
    srcs = glob(["**/*.java"]) + [
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:core_java_src",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:face_landmarker_java_src",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:image_segmenter_java_src",
    ],
    manifest = "AndroidManifest.xml",
    deps = [
        ":libmediapipe_tasks_vision_image_generator_jni_lib",
        "//mediapipe/framework:calculator_java_proto_lite",
        "//mediapipe/framework:calculator_options_java_proto_lite",
        "//mediapipe/framework/formats:classification_java_proto_lite",
        "//mediapipe/framework/formats:landmark_java_proto_lite",
        "//mediapipe/framework/formats:matrix_data_java_proto_lite",
        "//mediapipe/framework/formats:rect_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework_no_mff",
        "//mediapipe/java/com/google/mediapipe/framework/image",
        "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
        "//mediapipe/tasks/cc/core/proto:external_file_java_proto_lite",
        "//mediapipe/tasks/cc/vision/face_detector/proto:face_detector_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/face_geometry/proto:face_geometry_java_proto_lite",
        "//mediapipe/tasks/cc/vision/face_landmarker/proto:face_landmarker_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/face_landmarker/proto:face_landmarks_detector_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/image_generator/diffuser:stable_diffusion_iterate_calculator_java_proto_lite",
        "//mediapipe/tasks/cc/vision/image_generator/proto:conditioned_image_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/image_generator/proto:control_plugin_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/image_generator/proto:image_generator_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/image_segmenter/calculators:tensors_to_segmentation_calculator_java_proto_lite",
        "//mediapipe/tasks/cc/vision/image_segmenter/proto:image_segmenter_graph_options_java_proto_lite",
        "//mediapipe/tasks/cc/vision/image_segmenter/proto:segmenter_options_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:category",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:connection",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:normalized_landmark",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core:core_java",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:core_java",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:facelandmarker",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:imagesegmenter_java",
        "//third_party:any_java_proto",
        "//third_party:autovalue",
        "@com_google_protobuf//:protobuf_javalite",
        "@maven//:androidx_annotation_annotation",
        "@maven//:com_google_guava_guava",
    ],
)

mediapipe_build_aar_with_jni(
    name = "tasks_vision_image_generator",
    android_library = ":tasks_vision_image_generator_aar_library",
)
