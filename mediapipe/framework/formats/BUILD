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

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library")

package(
    default_visibility = ["//visibility:private"],
    features = ["-layering_check"],
)

proto_library(
    name = "detection_proto",
    srcs = ["detection.proto"],
    visibility = ["//mediapipe:__subpackages__"],
    deps = ["//mediapipe/framework/formats:location_data_proto"],
)

proto_library(
    name = "classification_proto",
    srcs = ["classification.proto"],
    visibility = ["//mediapipe:__subpackages__"],
)

proto_library(
    name = "image_format_proto",
    srcs = ["image_format.proto"],
    visibility = ["//mediapipe:__subpackages__"],
)

proto_library(
    name = "matrix_data_proto",
    srcs = ["matrix_data.proto"],
    visibility = ["//mediapipe:__subpackages__"],
)

proto_library(
    name = "location_data_proto",
    srcs = ["location_data.proto"],
    deps = ["//mediapipe/framework/formats/annotation:rasterization_proto"],
)

proto_library(
    name = "time_series_header_proto",
    srcs = ["time_series_header.proto"],
    visibility = ["//mediapipe:__subpackages__"],
)

mediapipe_cc_proto_library(
    name = "detection_cc_proto",
    srcs = ["detection.proto"],
    cc_deps = [":location_data_cc_proto"],
    visibility = ["//mediapipe:__subpackages__"],
    deps = [":detection_proto"],
)

mediapipe_cc_proto_library(
    name = "classification_cc_proto",
    srcs = ["classification.proto"],
    visibility = ["//mediapipe:__subpackages__"],
    deps = [":classification_proto"],
)

mediapipe_cc_proto_library(
    name = "image_format_cc_proto",
    srcs = ["image_format.proto"],
    visibility = [
        "//mediapipe:__subpackages__",
        "//mediapipe/java/com/google/mediapipe/framework:__subpackages__",
    ],
    deps = [":image_format_proto"],
)

mediapipe_cc_proto_library(
    name = "matrix_data_cc_proto",
    srcs = ["matrix_data.proto"],
    visibility = [
        "//mediapipe:__subpackages__",
        "//mediapipe/java/com/google/mediapipe/framework:__subpackages__",
    ],
    deps = [":matrix_data_proto"],
)

mediapipe_cc_proto_library(
    name = "location_data_cc_proto",
    srcs = ["location_data.proto"],
    cc_deps = ["//mediapipe/framework/formats/annotation:rasterization_cc_proto"],
    visibility = ["//mediapipe:__subpackages__"],
    deps = [":location_data_proto"],
)

mediapipe_cc_proto_library(
    name = "time_series_header_cc_proto",
    srcs = ["time_series_header.proto"],
    visibility = [
        "//mediapipe:__subpackages__",
        "//mediapipe/java/com/google/mediapipe/framework:__subpackages__",
    ],
    deps = [":time_series_header_proto"],
)

cc_library(
    name = "deleting_file",
    srcs = ["deleting_file.cc"],
    hdrs = ["deleting_file.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/port:logging",
    ],
)

cc_library(
    name = "matrix",
    srcs = ["matrix.cc"],
    hdrs = ["matrix.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:port",
        "//mediapipe/framework/formats:matrix_data_cc_proto",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@eigen_archive//:eigen",
    ],
)

cc_library(
    name = "image_frame",
    srcs = ["image_frame.cc"],
    hdrs = ["image_frame.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:port",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/port:aligned_malloc_and_free",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:source_location",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "image_frame_opencv",
    srcs = ["image_frame_opencv.cc"],
    hdrs = ["image_frame_opencv.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":image_frame",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/port:opencv_core",
    ],
)

cc_library(
    name = "location",
    srcs = ["location.cc"],
    hdrs = ["location.h"],
    defines = select({
        "//conditions:default": [],
        "//mediapipe:android": ["MEDIAPIPE_ANDROID_OPENCV"],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/formats/annotation:locus_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//mediapipe/framework:port",
        "//mediapipe/framework:type_map",
        "//mediapipe/framework/tool:status_util",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:point",
        "//mediapipe/framework/port:rectangle",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "//mediapipe/framework/formats/annotation:rasterization_cc_proto",
    ] + select({
        "//conditions:default": ["@com_google_protobuf//:protobuf"],
    }) + select({
        "//conditions:default": [
            "//mediapipe/framework/port:opencv_imgproc",
        ],
    }) + select({
        "//conditions:default": [
        ],
        "//mediapipe:android": [],
        "//mediapipe:apple": [],
    }),
    alwayslink = 1,
)

cc_library(
    name = "video_stream_header",
    hdrs = ["video_stream_header.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/formats:image_format_cc_proto",
    ],
)

cc_library(
    name = "yuv_image",
    hdrs = ["yuv_image.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/port:integral_types",
        "@libyuv",
    ],
)

cc_test(
    name = "image_frame_opencv_test",
    size = "small",
    srcs = ["image_frame_opencv_test.cc"],
    deps = [
        ":image_frame",
        ":image_frame_opencv",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
    ],
)

proto_library(
    name = "rect_proto",
    srcs = ["rect.proto"],
    visibility = ["//mediapipe:__subpackages__"],
)

mediapipe_cc_proto_library(
    name = "rect_cc_proto",
    srcs = ["rect.proto"],
    visibility = [
        "//mediapipe:__subpackages__",
    ],
    deps = [":rect_proto"],
)

proto_library(
    name = "landmark_proto",
    srcs = ["landmark.proto"],
    visibility = ["//mediapipe:__subpackages__"],
)

mediapipe_cc_proto_library(
    name = "landmark_cc_proto",
    srcs = ["landmark.proto"],
    visibility = ["//mediapipe:__subpackages__"],
    deps = [":landmark_proto"],
)
