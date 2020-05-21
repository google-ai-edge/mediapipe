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
# Description:
#  Port of various libraries and utilities.

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = ["//visibility:private"],
    features = ["-parse_headers"],
)

#TODO : remove from OSS.
config_setting(
    name = "use_mediapipe_threadpool",
    define_values = {
        "USE_MEDIAPIPE_THREADPOOL": "1",
    },
    visibility = ["//visibility:public"],
)

#TODO : remove from OSS.
config_setting(
    name = "use_google_threadpool",
    define_values = {
        "USE_MEDIAPIPE_THREADPOOL": "0",
    },
    visibility = ["//visibility:public"],
)

cc_library(
    name = "aligned_malloc_and_free",
    hdrs = ["aligned_malloc_and_free.h"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework/deps:aligned_malloc_and_free"],
)

# This proto lib is used by any code that needs to do proto I/O and proto template parsing.
cc_library(
    name = "advanced_proto",
    hdrs = [
        "advanced_proto_inc.h",
        "proto_ns.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":advanced_proto_lite",
        ":core_proto",
        "//mediapipe/framework:port",
    ] + select({
        "//conditions:default": ["@com_google_protobuf//:protobuf"],
    }),
)

cc_library(
    name = "advanced_proto_lite",
    hdrs = [
        "advanced_proto_lite_inc.h",
        "proto_ns.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":core_proto",
        "//mediapipe/framework:port",
    ] + select({
        "//conditions:default": ["@com_google_protobuf//:protobuf"],
    }),
)

cc_library(
    name = "any_proto",
    hdrs = ["any_proto.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":core_proto",
    ],
)

cc_library(
    name = "commandlineflags",
    hdrs = [
        "commandlineflags.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:port",
        "//third_party:glog",
    ],
)

cc_library(
    name = "core_proto",
    hdrs = [
        "core_proto_inc.h",
        "proto_ns.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:port",
    ] + select({
        "//conditions:default": ["@com_google_protobuf//:protobuf"],
    }),
)

cc_library(
    name = "file_helpers",
    hdrs = ["file_helpers.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":status",
        "//mediapipe/framework/deps:file_helpers",
        "//mediapipe/framework/deps:file_path",
    ],
)

cc_library(
    name = "image_resizer",
    hdrs = ["image_resizer.h"],
    visibility = ["//visibility:public"],
    deps = select({
        "//conditions:default": [
            "//mediapipe/framework/deps:image_resizer",
            "//mediapipe/framework/port:opencv_core",
        ],
    }),
)

cc_library(
    name = "integral_types",
    hdrs = ["integral_types.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "benchmark",
    testonly = 1,
    hdrs = ["benchmark.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_benchmark//:benchmark",
    ],
)

cc_library(
    name = "gtest_main",
    testonly = 1,
    hdrs = [
        "gmock.h",
        "gtest.h",
        "status_matchers.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:port",
        "//mediapipe/framework/deps:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "logging",
    hdrs = [
        "logging.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:port",
        "//third_party:glog",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "map_util",
    hdrs = [
        "map_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:port",
        "//mediapipe/framework/deps:map_util",
    ],
)

cc_library(
    name = "numbers",
    hdrs = ["numbers.h"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework/deps:numbers"],
)

cc_library(
    name = "opencv_core",
    hdrs = ["opencv_core_inc.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party:opencv",
    ],
)

cc_library(
    name = "opencv_imgproc",
    hdrs = ["opencv_imgproc_inc.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":opencv_core",
        "//third_party:opencv",
    ],
)

cc_library(
    name = "opencv_imgcodecs",
    hdrs = ["opencv_imgcodecs_inc.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":opencv_core",
        "//third_party:opencv",
    ],
)

cc_library(
    name = "opencv_highgui",
    hdrs = ["opencv_highgui_inc.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":opencv_core",
        "//third_party:opencv",
    ],
)

cc_library(
    name = "opencv_video",
    hdrs = ["opencv_video_inc.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":opencv_core",
        "//mediapipe/framework:port",
        "//third_party:opencv",
    ],
)

cc_library(
    name = "opencv_features2d",
    hdrs = ["opencv_features2d_inc.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":opencv_core",
        "//third_party:opencv",
    ],
)

cc_library(
    name = "opencv_calib3d",
    hdrs = ["opencv_calib3d_inc.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":opencv_core",
        "//third_party:opencv",
    ],
)

cc_library(
    name = "parse_text_proto",
    hdrs = [
        "parse_text_proto.h",
        "proto_ns.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":logging",
        "//mediapipe/framework:port",
    ] + select({
        "//conditions:default": ["@com_google_protobuf//:protobuf"],
    }),
)

cc_library(
    name = "point",
    hdrs = ["point2.h"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework/deps:point"],
)

cc_library(
    name = "port",
    hdrs = ["port.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:port",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "rectangle",
    hdrs = ["rectangle.h"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework/deps:rectangle"],
)

cc_library(
    name = "ret_check",
    hdrs = ["ret_check.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:port",
        "//mediapipe/framework/deps:ret_check",
    ],
)

cc_library(
    name = "singleton",
    hdrs = ["singleton.h"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework/deps:singleton"],
)

cc_library(
    name = "source_location",
    hdrs = [
        "source_location.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:port",
        "//mediapipe/framework/deps:source_location",
    ],
)

cc_library(
    name = "status",
    hdrs = [
        "canonical_errors.h",
        "status.h",
        "status_builder.h",
        "status_macros.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":source_location",
        "//mediapipe/framework:port",
        "//mediapipe/framework/deps:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "statusor",
    hdrs = [
        "statusor.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:port",
        "//mediapipe/framework/deps:statusor",
    ],
)

cc_library(
    name = "threadpool",
    hdrs = ["threadpool.h"],
    visibility = ["//visibility:public"],
    deps = select({
        "//conditions:default": [":threadpool_impl_default_to_google"],
        "//mediapipe:android": [":threadpool_impl_default_to_mediapipe"],
        "//mediapipe:apple": [":threadpool_impl_default_to_mediapipe"],
    }),
)

alias(
    name = "threadpool_impl_default_to_google",
    actual = select({
        "//conditions:default": "//mediapipe/framework/deps:threadpool",
        ":use_mediapipe_threadpool": "//mediapipe/framework/deps:threadpool",
    }),
)

alias(
    name = "threadpool_impl_default_to_mediapipe",
    actual = select({
        "//conditions:default": "//mediapipe/framework/deps:threadpool",
        ":use_google_threadpool": "//mediapipe/framework/deps:threadpool",
    }),
)

cc_library(
    name = "topologicalsorter",
    hdrs = ["topologicalsorter.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:port",
        "//mediapipe/framework/deps:topologicalsorter",
    ],
)

cc_library(
    name = "vector",
    hdrs = ["vector.h"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework/deps:vector"],
)
