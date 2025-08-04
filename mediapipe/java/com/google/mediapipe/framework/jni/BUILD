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

load("@rules_cc//cc:cc_library.bzl", "cc_library")

licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
    features = ["no_layering_check"],
)

alias(
    name = "mediapipe_android_framework_jni",
    actual = ":mediapipe_framework_jni",
)

cc_library(
    name = "mediapipe_framework_jni",
    srcs = [
        "compat_jni.cc",
        "graph.cc",
        "graph_jni.cc",
        "graph_profiler_jni.cc",
        "graph_service_jni.cc",
        "packet_context_jni.cc",
        "packet_creator_jni.cc",
        "packet_getter_jni.cc",
    ] + select({
        "//conditions:default": [],
        "//mediapipe:android": [
            "android_asset_util_jni.cc",
            "android_packet_creator_jni.cc",
            "resources_jni.cc",
        ],
    }) + select({
        "//conditions:default": [
            "graph_gl_sync_token.cc",
            "graph_texture_frame_jni.cc",
            "surface_output_jni.cc",
        ],
        "//mediapipe/gpu:disable_gpu": [],
    }),
    hdrs = [
        "colorspace.h",
        "compat_jni.h",
        "graph.h",
        "graph_jni.h",
        "graph_profiler_jni.h",
        "graph_service_jni.h",
        "packet_context_jni.h",
        "packet_creator_jni.h",
        "packet_getter_jni.h",
    ] + select({
        "//conditions:default": [],
        "//mediapipe:android": [
            "android_asset_util_jni.h",
            "android_packet_creator_jni.h",
            "resources_jni.h",
        ],
    }) + select({
        "//conditions:default": [
            "graph_gl_sync_token.h",
            "graph_texture_frame_jni.h",
            "surface_output_jni.h",
        ],
        "//mediapipe/gpu:disable_gpu": [],
    }),
    linkopts = select({
        "//conditions:default": [],
        "//mediapipe:android": [
            "-ljnigraphics",
            "-lEGL",  # This is needed by compat_jni even if GPU is disabled.
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":class_registry",
        ":jni_util",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_profile_cc_proto",
        "//mediapipe/framework:camera_intrinsics",
        "//mediapipe/framework:graph_service",
        "//mediapipe/framework:packet",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/formats:time_series_header_cc_proto",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:singleton",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:threadpool",
        "//mediapipe/framework/profiler:graph_profiler",
        "//mediapipe/framework/stream_handler:fixed_size_input_stream_handler",
        "//mediapipe/framework/tool:executor_util",
        "//mediapipe/framework/tool:name_util",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@eigen//:eigen3",
    ] + select({
        "//conditions:default": [
            "//mediapipe/framework/port:file_helpers",
        ],
        "//mediapipe:android": [
            "//mediapipe/util/android:asset_manager_util",
            "//mediapipe/util/android/file/base",
        ],
    }) + select({
        "//conditions:default": [
            "//mediapipe/gpu:egl_surface_holder",
            "//mediapipe/gpu:gl_base",
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gl_context",
            "//mediapipe/gpu:gl_quad_renderer",
            "//mediapipe/gpu:gl_surface_sink_calculator",
            "//mediapipe/gpu:gl_texture_buffer",
            "//mediapipe/gpu:gpu_buffer",
            "//mediapipe/gpu:gpu_shared_data_internal",
            "//mediapipe/gpu:graph_support",
        ],
        "//mediapipe/gpu:disable_gpu": [
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "jni_util",
    srcs = ["jni_util.cc"],
    hdrs = ["jni_util.h"],
    deps = [
        ":class_registry",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/synchronization",
    ] + select({
        "//conditions:default": [
        ],
        "//mediapipe:android": [
        ],
    }),
)

cc_library(
    name = "class_registry",
    srcs = ["class_registry.cc"],
    hdrs = ["class_registry.h"],
    deps = [
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ] + select({
        "//conditions:default": [
        ],
        "//mediapipe:android": [
        ],
    }),
)

cc_library(
    name = "register_natives",
    srcs = ["register_natives.cc"],
    hdrs = ["register_natives.h"],
    deps = [
        ":class_registry",
        ":loose_headers",
        ":mediapipe_framework_jni",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ] + select({
        "//conditions:default": [
        ],
        "//mediapipe:android": [
        ],
    }),
)

# For a more maintainable build this target should not exist and the headers
# should  be split into the existing cc_library targets, but this change was
# automatically  done so that we can remove long standing issues and complexity
# in the build system. It's up to the OWNERS of this package to get rid of it or
# not. The use of the textual_hdrs attribute is discouraged, use hdrs instead.
# Here it is used to avoid header parsing errors in packages where the feature
# parse_headers was enabled since loose headers were not being parsed.
cc_library(
    name = "loose_headers",
    tags = ["avoid_dep"],
    textual_hdrs = [
        "android_asset_util_jni.h",
        "android_packet_creator_jni.h",
        "resources_jni.h",
    ],
    visibility = ["//visibility:private"],
)
