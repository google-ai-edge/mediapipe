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

package(default_visibility = ["//mediapipe/framework:__subpackages__"])

# This is used to enable the profiler on platforms where it is not on by default.
# To enable, pass --define MEDIAPIPE_PROFILING=1 to bazel.
config_setting(
    name = "graph_profiler_enabled",
    values = {
        "define": "MEDIAPIPE_PROFILING=1",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "graph_profiler_disabled",
    values = {
        "define": "MEDIAPIPE_PROFILING=0",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_release",
    values = {
        "crosstool_top": "//tools/android:ndk",
        "compilation_mode": "opt",
        "define": "prod=1",
    },
    visibility = ["//visibility:public"],
)

# Maps either to graph_profiler_real or to graph_profiler_stub
# depending on the platform, compilation mode, and MEDIAPIPE_PROFILING value.
# This requires three levels of aliases because of limitations with select.
#
# Right now it must be explicitly enabled on mobile platforms, and
# it must be explictly disabled on other platforms including non-release
# android builds.
alias(
    name = "graph_profiler",
    actual = select({
        "//conditions:default": ":graph_profiler_default_on",
    }),
)

alias(
    name = "graph_profiler_default_off",
    actual = select({
        ":graph_profiler_enabled": ":graph_profiler_real",
        "//conditions:default": ":graph_profiler_stub",
    }),
    visibility = ["//visibility:private"],
)

alias(
    name = "graph_profiler_default_on",
    actual = select({
        ":graph_profiler_disabled": ":graph_profiler_stub",
        "//conditions:default": ":graph_profiler_real",
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "graph_profiler_stub",
    hdrs = ["graph_profiler_stub.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:status",
    ],
)

cc_library(
    name = "graph_profiler_real",
    srcs = [
        "graph_profiler.cc",
    ] + select({
        "//conditions:default": ["gl_context_profiler.cc"],
        "//mediapipe/gpu:disable_gpu": [],
    }),
    hdrs = [
        "graph_profiler.h",
    ],
    defines = [
        "MEDIAPIPE_PROFILER_AVAILABLE",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":profiler_resource_util",
        ":graph_tracer",
        ":trace_buffer",
        ":sharded_map",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_profile_cc_proto",
        "//mediapipe/framework/port:integral_types",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "//mediapipe/framework/deps:clock",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:executor",
        "//mediapipe/framework:validated_graph_config",
        "//mediapipe/framework/tool:tag_map",
        "//mediapipe/framework/tool:validate_name",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:re2",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:advanced_proto_lite",
        "//mediapipe/framework/tool:name_util",
    ] + select({
        "//conditions:default": [],
    }) + select({
        "//conditions:default": [
        ],
        "//mediapipe/gpu:disable_gpu": [],
    }),
)

cc_library(
    name = "circular_buffer",
    hdrs = ["circular_buffer.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/framework/port:integral_types",
    ],
)

cc_test(
    name = "circular_buffer_test",
    size = "small",
    srcs = ["circular_buffer_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":circular_buffer",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:threadpool",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "trace_buffer",
    srcs = ["trace_buffer.h"],
    hdrs = ["trace_buffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":circular_buffer",
        "//mediapipe/framework:calculator_profile_cc_proto",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:timestamp",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "graph_tracer",
    srcs = [
        "graph_tracer.cc",
        "trace_builder.cc",
        "trace_builder.h",
    ],
    hdrs = [
        "graph_tracer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":trace_buffer",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_profile_cc_proto",
        "//mediapipe/framework:input_stream_shard",
        "//mediapipe/framework:output_stream_shard",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:integral_types",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "sharded_map",
    hdrs = ["sharded_map.h"],
    visibility = ["//visibility:private"],
    deps = [
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "test_context_builder",
    testonly = 1,
    hdrs = ["test_context_builder.h"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework:mediapipe_options_cc_proto",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "//mediapipe/framework/tool:tag_map",
        "//mediapipe/framework/tool:tag_map_helper",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "graph_tracer_test",
    srcs = ["graph_tracer_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_profiler",
        ":graph_tracer",
        ":test_context_builder",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:immediate_mux_calculator",
        "//mediapipe/calculators/core:round_robin_demux_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_profile_cc_proto",
        "//mediapipe/framework:test_calculators",
        "//mediapipe/framework/deps:clock",
        "//mediapipe/framework/port:advanced_proto",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/stream_handler:default_input_stream_handler",
        "//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        "//mediapipe/framework/tool:simulation_clock",
        "//mediapipe/framework/tool:simulation_clock_executor",
        "//mediapipe/framework/tool:status_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "sharded_map_test",
    srcs = ["sharded_map_test.cc"],
    deps = [
        ":sharded_map",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:threadpool",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "profiler_resource_util",
    srcs = ["profiler_resource_util_common.cc"] + select({
        "//conditions:default": ["profiler_resource_util.cc"],
        "//mediapipe/framework:android_no_jni": ["profiler_resource_util_android_hal.cc"],
        "//mediapipe:android": ["profiler_resource_util_android.cc"],
        "//mediapipe:ios": ["profiler_resource_util_ios.cc"],
    }),
    hdrs = ["profiler_resource_util.h"],
    # We use Objective-C++ on iOS.
    copts = select({
        "//conditions:default": [],
        "//mediapipe:ios": [
            "-ObjC++",
        ],
    }),
    visibility = [
        "//mediapipe/framework:mediapipe_internal",
    ],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:statusor",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/deps:file_path",
    ] + select({
        "//conditions:default": [
            "//mediapipe/framework/port:file_helpers",
        ],
        "//mediapipe/framework:android_no_jni": [
            "//mediapipe/framework/port:file_helpers",
        ],
        "//mediapipe:android": [
            "//mediapipe/java/com/google/mediapipe/framework/jni:jni_util",
            "//mediapipe/framework/port:file_helpers",
        ],
        "//mediapipe:apple": [
            "//mediapipe/framework/port:file_helpers",
        ],
    }),
)

cc_test(
    name = "reporter_test",
    srcs = ["reporter_test.cc"],
    data = [
        "//mediapipe/framework/profiler/testdata:mediapipe_profile_graphs",
    ],
    visibility = ["//visibility:private"],
    deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_profile_cc_proto",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:advanced_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/profiler/reporter:reporter_lib",
        "//mediapipe/framework/tool:test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)
