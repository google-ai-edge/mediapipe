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
        "//mediapipe/graphs/hand_tracking:mobile_calculators",
        "//mediapipe/java/com/google/mediapipe/framework/jni:mediapipe_framework_jni",
    ],
)

cc_library(
    name = "mediapipe_jni_lib",
    srcs = [":libmediapipe_jni.so"],
    alwayslink = 1,
)

# Maps the binary graph to an alias (e.g., the app name) for convenience so that the alias can be
# easily incorporated into the app via, for example,
# MainActivity.BINARY_GRAPH_NAME = "appname.binarypb".
genrule(
    name = "binary_graph",
    srcs = ["//mediapipe/graphs/hand_tracking:hand_tracking_mobile_gpu_binary_graph"],
    outs = ["handtrackinggpu.binarypb"],
    cmd = "cp $< $@",
)

# To use the 3D model instead of the default 2D model, add "--define 3D=true" to the
# bazel build command.
config_setting(
    name = "use_3d_model",
    define_values = {
        "3D": "true",
    },
)

genrule(
    name = "model",
    srcs = select({
        "//conditions:default": ["//mediapipe/models:hand_landmark.tflite"],
        ":use_3d_model": ["//mediapipe/models:hand_landmark_3d.tflite"],
    }),
    outs = ["hand_landmark.tflite"],
    cmd = "cp $< $@",
)

android_library(
    name = "mediapipe_lib",
    srcs = glob(["*.java"]),
    assets = [
        ":binary_graph",
        ":model",
        "//mediapipe/models:palm_detection.tflite",
        "//mediapipe/models:palm_detection_labelmap.txt",
    ],
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
        "//third_party:androidx_material",
        "//third_party:androidx_recyclerview",
        "//third_party:opencv",
        "@androidx_concurrent_futures//jar",
        "@androidx_lifecycle//jar",
        "@com_google_code_findbugs//jar",
        "@com_google_guava_android//jar",
    ],
)

android_binary(
    name = "handtrackinggpu",
    manifest = "AndroidManifest.xml",
    manifest_values = {"applicationId": "com.google.mediapipe.apps.handtrackinggpu"},
    multidex = "native",
    deps = [
        ":mediapipe_lib",
    ],
)
