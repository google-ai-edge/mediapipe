# Copyright 2019 The MediaPipe Authors.
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

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:private"])

cc_binary(
    name = "libmediapipe_jni.so",
    linkshared = 1,
    linkstatic = 1,
    deps = [
        "//mediapipe/graphs/object_detection_3d:mobile_calculators",
        "//mediapipe/java/com/google/mediapipe/framework/jni:mediapipe_framework_jni",
    ],
)

cc_library(
    name = "mediapipe_jni_lib",
    srcs = [":libmediapipe_jni.so"],
    alwayslink = 1,
)

# To use the "chair" model instead of the default "shoes" model,
# add "--define chair=true" to the bazel build command.
config_setting(
    name = "use_chair_model",
    define_values = {
        "chair": "true",
    },
)

# Maps the binary graph to an alias (e.g., the app name) for convenience so that the alias can be
# easily incorporated into the app via, for example,
# MainActivity.BINARY_GRAPH_NAME = "appname.binarypb".
genrule(
    name = "binary_graph",
    srcs = select({
        "//conditions:default": ["//mediapipe/graphs/object_detection_3d:mobile_gpu_binary_graph_shoe"],
        ":use_chair_model": ["//mediapipe/graphs/object_detection_3d:mobile_gpu_binary_graph_chair"],
    }),
    outs = ["objectdetection3d.binarypb"],
    cmd = "cp $< $@",
)

genrule(
    name = "model",
    srcs = select({
        "//conditions:default": ["//mediapipe/models:object_detection_3d_sneakers.tflite"],
        ":use_chair_model": ["//mediapipe/models:object_detection_3d_chair.tflite"],
    }),
    outs = ["object_detection_3d.tflite"],
    cmd = "cp $< $@",
)

android_library(
    name = "mediapipe_lib",
    srcs = glob(["*.java"]),
    assets = [
        ":binary_graph",
        ":model",
        "//mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d/assets:box.obj.uuu",
        "//mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d/assets:classic_colors.png",
    ] + select({
        "//conditions:default": [
            "//mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d/assets/sneaker:model.obj.uuu",
            "//mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d/assets/sneaker:texture.bmp",
        ],
        ":use_chair_model": [
            "//mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d/assets/chair:model.obj.uuu",
            "//mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d/assets/chair:texture.bmp",
        ],
    }),
    assets_dir = "",
    manifest = "AndroidManifest.xml",
    resource_files = glob(["res/**"]),
    deps = [
        ":mediapipe_jni_lib",
        "//mediapipe/framework/formats:landmark_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/components:android_camerax_helper",
        "//mediapipe/java/com/google/mediapipe/components:android_components",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/glutil",
        "//third_party:androidx_appcompat",
        "//third_party:androidx_constraint_layout",
        "//third_party:androidx_legacy_support_v4",
        "//third_party:androidx_recyclerview",
        "//third_party:opencv",
        "@maven//:androidx_concurrent_concurrent_futures",
        "@maven//:androidx_lifecycle_lifecycle_common",
        "@maven//:com_google_code_findbugs_jsr305",
        "@maven//:com_google_guava_guava",
    ],
)

android_binary(
    name = "objectdetection3d",
    manifest = "AndroidManifest.xml",
    manifest_values = {"applicationId": "com.google.mediapipe.apps.objectdetection3d"},
    multidex = "native",
    deps = [
        ":mediapipe_lib",
    ],
)
