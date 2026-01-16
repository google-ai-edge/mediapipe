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

load("@bazel_skylib//lib:selects.bzl", "selects")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(":build_defs.bzl", "generate_manifest_values")

licenses(["notice"])

package(default_visibility = ["//visibility:private"])

config_setting(
    name = "use_chair",
    define_values = {
        "chair": "true",
    },
)

config_setting(
    name = "use_cup",
    define_values = {
        "cup": "true",
    },
)

config_setting(
    name = "use_camera",
    define_values = {
        "camera": "true",
    },
)

config_setting(
    name = "use_shoe_1stage",
    define_values = {
        "shoe_1stage": "true",
    },
)

config_setting(
    name = "use_chair_1stage",
    define_values = {
        "chair_1stage": "true",
    },
)

selects.config_setting_group(
    name = "1stage",
    match_any = [
        ":use_shoe_1stage",
        ":use_chair_1stage",
    ],
)

cc_binary(
    name = "libmediapipe_jni.so",
    linkshared = 1,
    linkstatic = 1,
    deps = select({
        "//conditions:default": ["//mediapipe/graphs/object_detection_3d:mobile_calculators"],
        ":1stage": ["//mediapipe/graphs/object_detection_3d:mobile_calculators_1stage"],
    }) + [
        "//mediapipe/java/com/google/mediapipe/framework/jni:mediapipe_framework_jni",
    ],
)

cc_library(
    name = "mediapipe_jni_lib",
    srcs = [":libmediapipe_jni.so"],
    alwayslink = 1,
)

genrule(
    name = "binary_graph",
    srcs = select({
        "//conditions:default": ["//mediapipe/graphs/object_detection_3d:mobile_gpu_binary_graph"],
        ":1stage": ["//mediapipe/graphs/object_detection_3d:mobile_gpu_1stage_binary_graph"],
    }),
    outs = ["object_detection_3d.binarypb"],
    cmd = "cp $< $@",
)

MODELS_DIR = "//mediapipe/modules/objectron"

genrule(
    name = "model",
    srcs = select({
        "//conditions:default": [MODELS_DIR + ":object_detection_3d_sneakers.tflite"],
        ":use_chair": [MODELS_DIR + ":object_detection_3d_chair.tflite"],
        ":use_cup": [MODELS_DIR + ":object_detection_3d_cup.tflite"],
        ":use_camera": [MODELS_DIR + ":object_detection_3d_camera.tflite"],
        ":use_shoe_1stage": [MODELS_DIR + ":object_detection_3d_sneakers_1stage.tflite"],
        ":use_chair_1stage": [MODELS_DIR + ":object_detection_3d_chair_1stage.tflite"],
    }),
    outs = ["object_detection_3d.tflite"],
    cmd = "cp $< $@",
)

MANIFESTS_DIR = "//mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d/manifests"

android_library(
    name = "manifest_lib",
    exports_manifest = 1,
    manifest = select({
        "//conditions:default": MANIFESTS_DIR + ":AndroidManifestSneaker.xml",
        ":use_chair": MANIFESTS_DIR + ":AndroidManifestChair.xml",
        ":use_cup": MANIFESTS_DIR + ":AndroidManifestCup.xml",
        ":use_camera": MANIFESTS_DIR + ":AndroidManifestCamera.xml",
        ":use_shoe_1stage": MANIFESTS_DIR + ":AndroidManifestSneaker.xml",
        ":use_chair_1stage": MANIFESTS_DIR + ":AndroidManifestChair.xml",
    }),
    deps = [
        "//third_party:opencv",
        "@maven//:androidx_concurrent_concurrent_futures",
        "@maven//:com_google_guava_guava",
    ],
)

ASSETS_DIR = "//mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d/assets"

genrule(
    name = "mesh",
    srcs = select({
        "//conditions:default": [ASSETS_DIR + "/sneaker:model.obj.uuu"],
        ":use_chair": [ASSETS_DIR + "/chair:model.obj.uuu"],
        ":use_cup": [ASSETS_DIR + "/cup:model.obj.uuu"],
        ":use_camera": [ASSETS_DIR + "/camera:model.obj.uuu"],
        ":use_shoe_1stage": [ASSETS_DIR + "/sneaker:model.obj.uuu"],
        ":use_chair_1stage": [ASSETS_DIR + "/chair:model.obj.uuu"],
    }),
    outs = ["model.obj.uuu"],
    cmd = "cp $< $@",
)

genrule(
    name = "texture",
    srcs = select({
        "//conditions:default": [ASSETS_DIR + "/sneaker:texture.jpg"],
        ":use_chair": [ASSETS_DIR + "/chair:texture.jpg"],
        ":use_cup": [ASSETS_DIR + "/cup:texture.jpg"],
        ":use_camera": [ASSETS_DIR + "/camera:texture.jpg"],
        ":use_shoe_1stage": [ASSETS_DIR + "/sneaker:texture.jpg"],
        ":use_chair_1stage": [ASSETS_DIR + "/chair:texture.jpg"],
    }),
    outs = ["texture.jpg"],
    cmd = "cp $< $@",
)

android_binary(
    name = "objectdetection3d",
    srcs = glob(["*.java"]),
    assets = [
        ":binary_graph",
        ":model",
        ":mesh",
        ":texture",
        MODELS_DIR + ":object_detection_ssd_mobilenetv2_oidv4_fp16.tflite",
        MODELS_DIR + ":object_detection_oidv4_labelmap.txt",
        ASSETS_DIR + ":box.obj.uuu",
        ASSETS_DIR + ":classic_colors.png",
    ],
    assets_dir = "",
    manifest = "//mediapipe/examples/android/src/java/com/google/mediapipe/apps/basic:AndroidManifest.xml",
    manifest_values = select({
        "//conditions:default": generate_manifest_values("com.google.mediapipe.apps.objectdetection3d_shoe", "Shoe Objectron"),
        ":use_chair": generate_manifest_values("com.google.mediapipe.apps.objectdetection3d_chair", "Chair Objectron"),
        ":use_cup": generate_manifest_values("com.google.mediapipe.apps.objectdetection3d_cup", "Cup Objectron"),
        ":use_camera": generate_manifest_values("com.google.mediapipe.apps.objectdetection3d_camera", "Camera Objectron"),
        ":use_shoe_1stage": generate_manifest_values("com.google.mediapipe.apps.objectdetection3d_shoe_1stage", "Single Stage Shoe Objectron"),
        ":use_chair_1stage": generate_manifest_values("com.google.mediapipe.apps.objectdetection3d_chair_1stage", "Single Stage Chair Objectron"),
    }),
    multidex = "native",
    deps = [
        ":manifest_lib",
        ":mediapipe_jni_lib",
        "//mediapipe/examples/android/src/java/com/google/mediapipe/apps/basic:basic_lib",
        "//mediapipe/framework/formats:landmark_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//third_party:libc++_shared_lib",
    ],
)
