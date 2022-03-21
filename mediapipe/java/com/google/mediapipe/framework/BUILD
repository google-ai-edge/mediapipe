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

licenses(["notice"])

# MediaPipe Android framework.

exports_files([
    "proguard.pgcfg",
])

android_library(
    name = "android_framework",
    proguard_specs = [":proguard.pgcfg"],
    visibility = ["//visibility:public"],
    exports = [
        ":android_core",
        ":android_framework_no_mff",
    ],
)

# TODO: Rename android_framework_no_mff.
android_library(
    name = "android_framework_no_mff",
    proguard_specs = [":proguard.pgcfg"],
    exports = [
        ":android_framework_no_proguard",
    ],
)

android_library(
    name = "android_framework_no_proguard",
    srcs = glob(
        ["Android*.java"],
    ) + [
        "AssetCache.java",
        "AssetCacheDbHelper.java",
        "MediaPipeRunner.java",
    ],
    visibility = [
        "//java/com/google/android/libraries/camera/effects:__subpackages__",
        "//mediapipe/java/com/google/mediapipe:__subpackages__",
    ],
    exports = [
        ":android_core",
    ],
    deps = [
        ":android_core",
        "//third_party:androidx_annotation",
        "//third_party:androidx_legacy_support_v4",
        "@maven//:com_google_code_findbugs_jsr305",
        "@maven//:com_google_flogger_flogger",
        "@maven//:com_google_flogger_flogger_system_backend",
        "@maven//:com_google_guava_guava",
    ],
)

# This is the Android version of "framework".
# TODO: unify once allowed by bazel.
# Note: this is not called "android_framework" for historical reasons (that target
# also includes other libraries).
android_library(
    name = "android_core",
    srcs = glob(
        ["**/*.java"],
        exclude = [
            "Android*",
            "AssetCache.java",
            "AssetCacheDbHelper.java",
            "MediaPipeRunner.java",
            "ProtoUtil.java",
            "TypeNameRegistry.java",
            "TypeNameRegistryLite.java",
            "TypeNameRegistryFull.java",
            "MediaPipeException.java",
        ],
    ),
    exports = [
        ":framework_proto_lite",
        ":mediapipe_exception_android",
    ],
    deps = [
        ":framework_proto_lite",
        ":mediapipe_exception_android",
        "@com_google_protobuf//:protobuf_javalite",
        "@maven//:com_google_code_findbugs_jsr305",
        "@maven//:com_google_flogger_flogger",
        "@maven//:com_google_flogger_flogger_system_backend",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "framework_proto_lite",
    srcs = [
        "ProtoUtil.java",
        "TypeNameRegistry.java",
        "TypeNameRegistryLite.java",
    ],
    exports = [
        "//mediapipe/framework:calculator_java_proto_lite",
        "//mediapipe/framework:calculator_profile_java_proto_lite",
        "//mediapipe/framework:stream_handler_java_proto_lite",
        "//mediapipe/framework/tool:calculator_graph_template_java_proto_lite",
    ],
    deps = [
        ":mediapipe_exception_android",
        "//mediapipe/framework:calculator_java_proto_lite",
        "//mediapipe/framework:calculator_profile_java_proto_lite",
        "//mediapipe/framework:stream_handler_java_proto_lite",
        "//mediapipe/framework/tool:calculator_graph_template_java_proto_lite",
        "@com_google_protobuf//:protobuf_javalite",
        "@maven//:com_google_guava_guava",
    ],
)

android_library(
    name = "mediapipe_exception_android",
    srcs = [
        "MediaPipeException.java",
    ],
    deps = [
        "@maven//:com_google_guava_guava",
    ],
)

# Expose the java source files for building mediapipe AAR.
filegroup(
    name = "java_src",
    srcs = glob(
        ["*.java"],
        exclude = ["TypeNameRegistryFull.java"],
    ),
    visibility = ["//mediapipe:__subpackages__"],
)
