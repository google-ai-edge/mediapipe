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
#   The dependencies of mediapipe.

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])

package_group(
    name = "mediapipe_internal",
    packages = [
        "//mediapipe/...",
    ],
)

package(default_visibility = ["mediapipe_internal"])

bzl_library(
    name = "expand_template_bzl",
    srcs = [
        "expand_template.bzl",
    ],
    visibility = ["//mediapipe/framework:__subpackages__"],
)

bzl_library(
    name = "descriptor_set_bzl",
    srcs = [
        "descriptor_set.bzl",
    ],
    visibility = ["//mediapipe/framework:__subpackages__"],
)

mediapipe_proto_library(
    name = "proto_descriptor_proto",
    srcs = ["proto_descriptor.proto"],
    visibility = [
        "//mediapipe/deps:__subpackages__",
        "//mediapipe/framework:__subpackages__",
    ],
)

cc_library(
    name = "aligned_malloc_and_free",
    hdrs = ["aligned_malloc_and_free.h"],
)

cc_library(
    name = "cleanup",
    hdrs = ["cleanup.h"],
    deps = ["@com_google_absl//absl/base:core_headers"],
)

cc_library(
    name = "clock",
    srcs = [
        "clock.cc",
        "monotonic_clock.cc",
    ],
    hdrs = [
        "clock.h",
        "monotonic_clock.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "message_matchers",
    testonly = True,
    hdrs = ["message_matchers.h"],

    # Use this library through "mediapipe/framework/port:gtest_main".
    visibility = [
        "//mediapipe/framework/port:__pkg__",
    ],
    deps = [
        "//mediapipe/framework/port:core_proto",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "file_path",
    srcs = ["file_path.cc"],
    hdrs = ["file_path.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "file_helpers",
    srcs = ["file_helpers.cc"],
    hdrs = ["file_helpers.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":file_path",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "intops",
    hdrs = [
        "safe_int.h",
        "strong_int.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/port",
        "//mediapipe/framework/port:integral_types",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
    ],
)

cc_library(
    name = "image_resizer",
    hdrs = ["image_resizer.h"],
    deps = [
        "//mediapipe/framework/port:opencv_imgproc",
    ],
)

cc_library(
    name = "map_util",
    hdrs = ["map_util.h"],

    # Use this library through "mediapipe/framework/port:map_util".
    visibility = ["//mediapipe/framework/port:__pkg__"],
    deps = [
        "//mediapipe/framework/port:logging",
        "@com_google_absl//absl/log:absl_check",
    ],
)

cc_library(
    name = "mathutil",
    hdrs = ["mathutil.h"],
    visibility = [
        "//mediapipe:__subpackages__",
    ],
    deps = [
        "//mediapipe/framework/port:integral_types",
        "@com_google_absl//absl/log:absl_check",
    ],
)

cc_library(
    name = "numbers",
    hdrs = ["numbers.h"],
    visibility = ["//mediapipe/framework/port:__pkg__"],
    deps = [
        "//mediapipe/framework/port:integral_types",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "no_destructor",
    hdrs = ["no_destructor.h"],
)

cc_library(
    name = "point",
    hdrs = ["point2.h"],

    # Use this library through "mediapipe/framework/port:point".
    visibility = ["//mediapipe/framework/port:__pkg__"],
    deps = [
        ":mathutil",
        ":vector",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
    ],
)

cc_library(
    name = "random",
    hdrs = ["random_base.h"],
    deps = ["//mediapipe/framework/port:integral_types"],
)

cc_library(
    name = "rectangle",
    hdrs = ["rectangle.h"],

    # Use this library through "mediapipe/framework/port:rectangle".
    visibility = ["//mediapipe/framework/port:__pkg__"],
    deps = [
        ":point",
        ":vector",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
    ],
)

cc_library(
    name = "registration_token",
    srcs = ["registration_token.cc"],
    hdrs = ["registration_token.h"],
)

cc_library(
    name = "registration",
    srcs = ["registration.cc"],
    hdrs = ["registration.h"],
    visibility = [
        "mediapipe_internal",
        "//buzz/diffractor/mediapipe:__pkg__",
    ],
    deps = [
        ":registration_token",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "singleton",
    hdrs = ["singleton.h"],

    # Use this library through "mediapipe/framework/port:singleton".
    visibility = ["//mediapipe/framework/port:__pkg__"],
    deps = [
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "source_location",
    hdrs = ["source_location.h"],

    # Use this library through "mediapipe/framework/port:source_location".
    visibility = ["//mediapipe/framework/port:__pkg__"],
)

cc_library(
    name = "status",
    srcs = [
        "status.cc",
        "status_builder.cc",
    ],
    hdrs = [
        "canonical_errors.h",
        "status.h",
        "status_builder.h",
        "status_macros.h",
    ],

    # Use this library through "mediapipe/framework/port:status".
    visibility = ["//mediapipe/framework/port:__pkg__"],
    deps = [
        ":source_location",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "re2",
    hdrs = [
        "re2.h",
    ],
)

cc_library(
    name = "status_matchers",
    testonly = True,
    hdrs = ["status_matchers.h"],

    # Use this library through "mediapipe/framework/port:gtest_main".
    visibility = ["//mediapipe/framework/port:__pkg__"],
    deps = [
        "//mediapipe/framework/port:statusor",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "requires",
    hdrs = ["requires.h"],

    # Use this library through "mediapipe/framework/port:requires".
    visibility = ["//mediapipe/framework/port:__pkg__"],
)

cc_library(
    name = "ret_check",
    srcs = ["ret_check.cc"],
    hdrs = ["ret_check.h"],

    # Use this library through "mediapipe/framework/port:ret_check".
    visibility = ["//mediapipe/framework/port:__pkg__"],
    deps = [
        ":status",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "thread_options",
    hdrs = ["thread_options.h"],
)

cc_library(
    name = "threadpool",
    srcs = select({
        "//mediapipe:windows": ["threadpool_std_thread_impl.cc"],
        "//conditions:default": ["threadpool_pthread_impl.cc"],
    }),
    hdrs = ["threadpool.h"],

    # Use this library through "mediapipe/framework/port:threadpool".
    visibility = ["//mediapipe/framework/port:__pkg__"],
    deps = [
        ":thread_options",
        "//mediapipe/framework/port:logging",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "topologicalsorter",
    srcs = ["topologicalsorter.cc"],
    hdrs = ["topologicalsorter.h"],

    # Use this library through "mediapipe/framework/port:topologicalsorter".
    visibility = ["//mediapipe/framework/port:__pkg__"],
    deps = [
        "//mediapipe/framework/port:logging",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
    ],
)

cc_library(
    name = "vector",
    hdrs = ["vector.h"],

    # Use this library through "mediapipe/framework/port:vector".
    visibility = ["//mediapipe/framework/port:__pkg__"],
    deps = [
        "//mediapipe/framework/port:integral_types",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/utility",
    ],
)

cc_test(
    name = "mathutil_unittest",
    srcs = ["mathutil_unittest.cc"],
    deps = [
        ":mathutil",
        "//mediapipe/framework/port:benchmark",
        "//mediapipe/framework/port:gtest_main",
    ],
)

cc_test(
    name = "registration_token_test",
    srcs = ["registration_token_test.cc"],
    linkstatic = 1,
    deps = [
        ":registration_token",
        "//mediapipe/framework/port:gtest_main",
    ],
)

cc_test(
    name = "safe_int_test",
    size = "small",
    timeout = "long",
    srcs = ["safe_int_test.cc"],
    linkstatic = 1,
    deps = [
        ":intops",
        "//mediapipe/framework/port:gtest_main",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_test(
    name = "monotonic_clock_test",
    srcs = ["monotonic_clock_test.cc"],
    linkstatic = 1,
    deps = [
        ":clock",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:threadpool",
        "//mediapipe/framework/tool:simulation_clock",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "status_builder_test",
    size = "small",
    srcs = ["status_builder_test.cc"],
    linkstatic = 1,
    deps = [
        ":status",
        "//mediapipe/framework/port:gtest_main",
    ],
)

cc_test(
    name = "status_test",
    size = "small",
    srcs = ["status_test.cc"],
    linkstatic = 1,
    deps = [
        ":status",
        ":status_matchers",
        "//mediapipe/framework/port:gtest_main",
    ],
)

cc_test(
    name = "topologicalsorter_test",
    srcs = ["topologicalsorter_test.cc"],
    linkstatic = 1,
    deps = [
        ":topologicalsorter",
        "//mediapipe/framework/port:gtest_main",
    ],
)

cc_test(
    name = "threadpool_test",
    srcs = ["threadpool_test.cc"],
    linkstatic = 1,
    deps = [
        ":threadpool",
        "//mediapipe/framework/port:gtest_main",
        "@com_google_absl//absl/synchronization",
    ],
)
