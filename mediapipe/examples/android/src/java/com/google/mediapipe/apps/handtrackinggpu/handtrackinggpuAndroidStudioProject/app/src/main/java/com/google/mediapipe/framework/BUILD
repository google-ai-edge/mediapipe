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

# MediaPipe Android framework.

exports_files(["proguard.pgcfg"])

android_library(
    name = "android_framework",
    proguard_specs = [
        ":proguard.pgcfg",
    ],
    visibility = ["//visibility:public"],
    exports = [
        ":android_core",
        ":android_framework_no_mff",
    ],
)

# TODO: Rename android_framework_no_mff.
android_library(
    name = "android_framework_no_mff",
    srcs = glob(
        ["Android*.java"],
    ) + [
        "AssetCache.java",
        "AssetCacheDbHelper.java",
        "MediaPipeRunner.java",
    ],
    proguard_specs = [
        ":proguard.pgcfg",
    ],
    exports = [
        ":android_core",
    ],
    deps = [
        ":android_core",
        "//third_party:androidx_annotation",
        "//third_party:androidx_legacy_support_v4",
        "@com_google_code_findbugs//jar",
        "@com_google_common_flogger//jar",
        "@com_google_common_flogger_system_backend//jar",
        "@com_google_guava_android//jar",
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
        ],
    ),
    deps = [
        "//mediapipe/framework:calculator_java_proto_lite",
        "//mediapipe/framework:calculator_profile_java_proto_lite",
        "//mediapipe/framework/tool:calculator_graph_template_java_proto_lite",
        "@com_google_code_findbugs//jar",
        "@com_google_common_flogger//jar",
        "@com_google_common_flogger_system_backend//jar",
        "@com_google_guava_android//jar",
    ],
)
