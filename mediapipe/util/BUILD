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
#

licenses(["notice"])

package(default_visibility = ["//visibility:private"])

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library")

proto_library(
    name = "audio_decoder_proto",
    srcs = ["audio_decoder.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

proto_library(
    name = "color_proto",
    srcs = ["color.proto"],
    visibility = ["//visibility:public"],
)

proto_library(
    name = "render_data_proto",
    srcs = ["render_data.proto"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/util:color_proto"],
)

mediapipe_cc_proto_library(
    name = "color_cc_proto",
    srcs = ["color.proto"],
    visibility = ["//visibility:public"],
    deps = [":color_proto"],
)

mediapipe_cc_proto_library(
    name = "render_data_cc_proto",
    srcs = ["render_data.proto"],
    cc_deps = [":color_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":render_data_proto"],
)

mediapipe_cc_proto_library(
    name = "audio_decoder_cc_proto",
    srcs = ["audio_decoder.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":audio_decoder_proto"],
)

cc_library(
    name = "audio_decoder",
    srcs = ["audio_decoder.cc"],
    hdrs = ["audio_decoder.h"],
    visibility = ["//mediapipe:__subpackages__"],
    deps = [
        ":audio_decoder_cc_proto",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/deps:cleanup",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/formats:time_series_header_cc_proto",
        "//mediapipe/framework/port:commandlineflags",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:map_util",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:status_util",
        "//third_party:libffmpeg",
        "@com_google_absl//absl/base:endian",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@eigen_archive//:eigen",
    ],
)

cc_library(
    name = "cpu_util",
    srcs = ["cpu_util.cc"],
    hdrs = ["cpu_util.h"],
    visibility = [
        "//mediapipe/framework:mediapipe_internal",
    ],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework:port",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
    ] + select({
        "//conditions:default": [],
        "//mediapipe:android": [
            "@androidndk//:cpufeatures",
        ],
    }),
)

cc_library(
    name = "header_util",
    srcs = ["header_util.cc"],
    hdrs = ["header_util.h"],
    visibility = [
        "//mediapipe/framework:mediapipe_internal",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:packet_set",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:status_util",
    ],
)

cc_library(
    name = "image_frame_util",
    srcs = ["image_frame_util.cc"],
    hdrs = ["image_frame_util.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/framework/deps:mathutil",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:yuv_image",
        "//mediapipe/framework/port",
        "//mediapipe/framework/port:aligned_malloc_and_free",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:status_util",
        "@com_google_absl//absl/strings",
        "@libyuv",
    ],
)

cc_library(
    name = "annotation_renderer",
    srcs = ["annotation_renderer.cc"],
    hdrs = ["annotation_renderer.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":render_data_cc_proto",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:vector",
        "//mediapipe/util:color_cc_proto",
    ],
)

cc_library(
    name = "resource_util",
    srcs = select({
        "//conditions:default": ["resource_util.cc"],
        "//mediapipe:android": ["resource_util_android.cc"],
        "//mediapipe:ios": ["resource_util_apple.cc"],
        "//mediapipe:macos": ["resource_util.cc"],
    }),
    hdrs = ["resource_util.h"],
    # We use Objective-C++ on iOS.
    copts = select({
        "//conditions:default": [],
        "//mediapipe:ios": [
            "-ObjC++",
        ],
        "//mediapipe:macos": [],
    }),
    visibility = [
        "//mediapipe/framework:mediapipe_internal",
    ],
    deps = [
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "@com_google_absl//absl/strings",
    ] + select({
        "//conditions:default": [
            "//mediapipe/framework/deps:file_path",
            "//mediapipe/framework/port:file_helpers",
            "@com_google_absl//absl/flags:flag",
        ],
        "//mediapipe:android": [
            "//mediapipe/util/android:asset_manager_util",
            "//mediapipe/util/android/file/base",
        ],
        "//mediapipe:ios": [],
        "//mediapipe:macos": [
            "//mediapipe/framework/deps:file_path",
            "//mediapipe/framework/port:file_helpers",
            "@com_google_absl//absl/flags:flag",
        ],
    }),
)

cc_library(
    name = "tensor_to_detection",
    srcs = ["tensor_to_detection.cc"],
    hdrs = ["tensor_to_detection.h"],
    features = [
        # Layering check doesn't play nicely with portable proto wrappers.
        "no_layering_check",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:map_util",
    ] + select({
        "//conditions:default": [
            "@org_tensorflow//tensorflow/core:framework",
        ],
        "//mediapipe:android": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
        ],
    }),
)

cc_library(
    name = "time_series_util",
    srcs = ["time_series_util.cc"],
    hdrs = ["time_series_util.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/formats:time_series_header_cc_proto",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "time_series_test_util",
    testonly = 1,
    hdrs = ["time_series_test_util.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":time_series_util",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/deps:message_matchers",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/formats:time_series_header_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen",
    ],
)

cc_test(
    name = "time_series_util_test",
    size = "small",
    srcs = ["time_series_util_test.cc"],
    deps = [
        ":time_series_util",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/deps:message_matchers",
        "//mediapipe/framework/formats:time_series_header_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "@eigen_archive//:eigen",
    ],
)
