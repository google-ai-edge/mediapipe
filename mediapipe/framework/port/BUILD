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

licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
    features = ["-parse_headers"],
)

#TODO : remove from OSS.
config_setting(
    name = "use_mediapipe_threadpool",
    define_values = {
        "USE_MEDIAPIPE_THREADPOOL": "1",
    },
)

#TODO : remove from OSS.
config_setting(
    name = "use_google_threadpool",
    define_values = {
        "USE_MEDIAPIPE_THREADPOOL": "0",
    },
)

cc_library(
    name = "aligned_malloc_and_free",
    hdrs = ["aligned_malloc_and_free.h"],
    deps = [
        "//mediapipe/framework/deps:aligned_malloc_and_free",
        "@com_google_absl//absl/base:core_headers",
    ],
)

# This proto lib is used by any code that needs to do proto I/O and proto template parsing.
cc_library(
    name = "advanced_proto",
    hdrs = [
        "advanced_proto_inc.h",
        "proto_ns.h",
    ],
    deps = [
        ":advanced_proto_lite",
        ":core_proto",
        "//mediapipe/framework:port",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "advanced_proto_lite",
    hdrs = [
        "advanced_proto_lite_inc.h",
        "proto_ns.h",
    ],
    deps = [
        ":core_proto",
        "//mediapipe/framework:port",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "any_proto",
    hdrs = ["any_proto.h"],
    deps = [
        ":core_proto",
    ],
)

cc_library(
    name = "commandlineflags",
    hdrs = [
        "commandlineflags.h",
    ],
    deps = [
        "//third_party:glog",
        "@com_google_absl//absl/flags:flag",
    ],
)

cc_library(
    name = "core_proto",
    hdrs = [
        "core_proto_inc.h",
        "proto_ns.h",
    ],
    deps = [
        "//mediapipe/framework:port",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "file_helpers",
    hdrs = ["file_helpers.h"],
    deps = [
        ":status",
        "//mediapipe/framework/deps:file_helpers",
        "//mediapipe/framework/deps:file_path",
    ],
)

cc_library(
    name = "image_resizer",
    hdrs = ["image_resizer.h"],
    deps = select({
        "//conditions:default": [
            "//mediapipe/framework/deps:image_resizer",
            "//mediapipe/framework/port:opencv_core",
        ],
        "//mediapipe:emscripten": ["//mediapipe/framework/deps:image_resizer"],
    }),
)

cc_library(
    name = "integral_types",
    hdrs = ["integral_types.h"],
)

cc_library(
    name = "benchmark",
    testonly = True,
    hdrs = ["benchmark.h"],
    deps = [
        "@com_google_benchmark//:benchmark",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "re2",
    hdrs = [
        "re2.h",
    ],
    deps = [
        "//mediapipe/framework/deps:re2",
    ],
)

cc_library(
    name = "gtest",
    testonly = True,
    hdrs = [
        "gmock.h",
        "gtest.h",
        "gtest-spi.h",
        "status_matchers.h",
    ],
    deps = [
        ":status_matchers",
        "//mediapipe/framework/deps:message_matchers",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "gtest_main",
    testonly = True,
    hdrs = [
        "gmock.h",
        "gtest.h",
        "gtest-spi.h",
        "status_matchers.h",
    ],
    deps = [
        ":status_matchers",
        "//mediapipe/framework/deps:message_matchers",
        "//mediapipe/framework/deps:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gtest_main_no_heapcheck",
    testonly = True,
    hdrs = [
        "gmock.h",
        "gtest.h",
        "gtest-spi.h",
        "status_matchers.h",
    ],
    deps = [
        ":status_matchers",
        "//mediapipe/framework/deps:message_matchers",
        "//mediapipe/framework/deps:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "logging",
    hdrs = [
        "logging.h",
    ],
    deps = [
        "//mediapipe/framework:port",
        "//third_party:glog",
        "@com_google_absl//absl/strings:has_ostream_operator",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "map_util",
    hdrs = [
        "map_util.h",
    ],
    deps = [
        "//mediapipe/framework:port",
        "//mediapipe/framework/deps:map_util",
    ],
)

cc_library(
    name = "numbers",
    hdrs = ["numbers.h"],
    deps = ["//mediapipe/framework/deps:numbers"],
)

# Disabling opencv when defining MEDIAPIPE_DISABLE_OPENCV to 1 in the bazel command.
# Note that this only applies to a select few calculators/framework components currently.
config_setting(
    name = "disable_opencv",
    define_values = {
        "MEDIAPIPE_DISABLE_OPENCV": "1",
    },
)

# Enable Halide when defining MEDIAPIPE_ENABLE_HALIDE to 1 in bazel command.
# This will  automatically disable opencv, i.e. will set MEDIAPIPE_DISABLE_OPENCV to 1.
# Note that this only applies to a select few calculators/framework components currently.
# TODO: Improve this. This only sets MEDIAPIPE_DISABLE_OPENCV as a "defines" Make value,
# not as a bazel "--define" variable, which has effects in C++ code but not in select() statements.
config_setting(
    name = "enable_halide",
    define_values = {
        "MEDIAPIPE_ENABLE_HALIDE": "1",
    },
)

cc_library(
    name = "opencv_core",
    hdrs = ["opencv_core_inc.h"],
    deps = [
        "//third_party:opencv",
    ],
)

cc_library(
    name = "opencv_imgproc",
    hdrs = ["opencv_imgproc_inc.h"],
    deps = [
        ":opencv_core",
        "//third_party:opencv",
    ],
)

cc_library(
    name = "opencv_imgcodecs",
    hdrs = ["opencv_imgcodecs_inc.h"],
    deps = [
        ":opencv_core",
        "//third_party:opencv",
    ],
)

cc_library(
    name = "opencv_photo",
    hdrs = ["opencv_photo_inc.h"],
    deps = [
        ":opencv_core",
        "//third_party:opencv",
    ],
)

cc_library(
    name = "opencv_video",
    hdrs = ["opencv_video_inc.h"],
    deps = [
        ":opencv_core",
        "//mediapipe/framework:port",
        "//third_party:opencv",
    ],
)

cc_library(
    name = "opencv_features2d",
    hdrs = ["opencv_features2d_inc.h"],
    deps = [
        ":opencv_core",
        "//third_party:opencv",
    ],
)

cc_library(
    name = "opencv_calib3d",
    hdrs = ["opencv_calib3d_inc.h"],
    deps = [
        ":opencv_core",
        "//third_party:opencv",
    ],
)

cc_library(
    name = "opencv_highgui",
    hdrs = ["opencv_highgui_inc.h"],
    deps = [
        ":opencv_core",
        "//third_party:opencv",
    ],
)

cc_library(
    name = "opencv_videoio",
    hdrs = ["opencv_videoio_inc.h"],
    deps = [
        ":opencv_core",
        "//mediapipe/framework:port",
        "//third_party:opencv",
    ],
)

cc_library(
    name = "parse_text_proto",
    hdrs = [
        "parse_text_proto.h",
        "proto_ns.h",
    ],
    deps = [
        ":core_proto",
        ":logging",
        "//mediapipe/framework:port",
        "@com_google_absl//absl/log:absl_check",
    ],
)

cc_library(
    name = "point",
    hdrs = ["point2.h"],
    deps = ["//mediapipe/framework/deps:point"],
)

cc_library(
    name = "port",
    hdrs = ["port.h"],
    deps = [
        "//mediapipe/framework:port",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "rectangle",
    hdrs = ["rectangle.h"],
    deps = ["//mediapipe/framework/deps:rectangle"],
)

cc_library(
    name = "ret_check",
    hdrs = ["ret_check.h"],
    deps = [
        "//mediapipe/framework:port",
        "//mediapipe/framework/deps:ret_check",
    ],
)

cc_library(
    name = "singleton",
    hdrs = ["singleton.h"],
    deps = ["//mediapipe/framework/deps:singleton"],
)

cc_library(
    name = "source_location",
    hdrs = [
        "source_location.h",
    ],
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
    deps = [
        "//mediapipe/framework:port",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "status_matchers",
    testonly = True,
    hdrs = ["status_matchers.h"],
    deps = [
        ":status",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "threadpool",
    hdrs = ["threadpool.h"],
    deps = select({
        "//conditions:default": [":threadpool_impl_default_to_google"],
        "//mediapipe:android": [":threadpool_impl_default_to_mediapipe"],
        "//mediapipe:apple": [":threadpool_impl_default_to_mediapipe"],
        "//mediapipe:emscripten": [":threadpool_impl_default_to_mediapipe"],
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
    deps = [
        "//mediapipe/framework:port",
        "//mediapipe/framework/deps:topologicalsorter",
    ],
)

cc_library(
    name = "vector",
    hdrs = ["vector.h"],
    deps = ["//mediapipe/framework/deps:vector"],
)

cc_library(
    name = "requires",
    hdrs = ["requires.h"],
    deps = ["//mediapipe/framework/deps:requires"],
)
