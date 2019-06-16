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

package(default_visibility = ["//visibility:private"])

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library")

proto_library(
    name = "packet_resampler_calculator_proto",
    srcs = ["packet_resampler_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework:calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "packet_resampler_calculator_cc_proto",
    srcs = ["packet_resampler_calculator.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//mediapipe:__subpackages__"],
    deps = [":packet_resampler_calculator_proto"],
)

cc_library(
    name = "counting_source_calculator",
    srcs = ["counting_source_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "make_pair_calculator",
    srcs = ["make_pair_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mux_calculator",
    srcs = ["mux_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/stream_handler:mux_input_stream_handler",
    ],
    alwayslink = 1,
)

cc_library(
    name = "packet_cloner_calculator",
    srcs = ["packet_cloner_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "pass_through_calculator",
    srcs = ["pass_through_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "round_robin_demux_calculator",
    srcs = ["round_robin_demux_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
    ],
    alwayslink = 1,
)

cc_library(
    name = "immediate_mux_calculator",
    srcs = ["immediate_mux_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "previous_loopback_calculator",
    srcs = ["previous_loopback_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/stream_handler:immediate_input_stream_handler",
    ],
    alwayslink = 1,
)

cc_library(
    name = "real_time_flow_limiter_calculator",
    srcs = ["real_time_flow_limiter_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        "//mediapipe/util:header_util",
    ],
    alwayslink = 1,
)

cc_test(
    name = "immediate_mux_calculator_test",
    srcs = ["immediate_mux_calculator_test.cc"],
    deps = [
        ":immediate_mux_calculator",
        ":round_robin_demux_calculator",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:test_calculators",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:threadpool",
        "//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        "//mediapipe/framework/tool:sink",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "packet_resampler_calculator",
    srcs = ["packet_resampler_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/calculators/core:packet_resampler_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:collection_item_id",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework/tool:options_util",
        "@com_google_absl//absl/strings",
        "//mediapipe/framework/deps:mathutil",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:integral_types",
    ] + select({
        "//conditions:default": [
            "//mediapipe/framework/deps:random",
        ],
    }),
    alwayslink = 1,
)

cc_test(
    name = "packet_resampler_calculator_test",
    timeout = "short",
    srcs = ["packet_resampler_calculator_test.cc"],
    deps = [
        ":packet_resampler_calculator",
        "//mediapipe/calculators/core:packet_resampler_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "previous_loopback_calculator_test",
    srcs = ["previous_loopback_calculator_test.cc"],
    deps = [
        ":previous_loopback_calculator",
        "//mediapipe/calculators/core:make_pair_calculator",
        "//mediapipe/calculators/core:pass_through_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        "//mediapipe/framework/tool:sink",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "real_time_flow_limiter_calculator_test",
    srcs = ["real_time_flow_limiter_calculator_test.cc"],
    deps = [
        ":real_time_flow_limiter_calculator",
        "//mediapipe/calculators/core:counting_source_calculator",
        "//mediapipe/calculators/core:pass_through_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:test_calculators",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        "//mediapipe/framework/tool:sink",
        "@com_google_absl//absl/time",
    ],
)
