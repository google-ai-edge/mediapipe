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

load("@bazel_skylib//lib:selects.bzl", "selects")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

selects.config_setting_group(
    name = "apple_or_disable_gpu",
    match_any = [
        "//mediapipe/gpu:disable_gpu",
        "//mediapipe:apple",
    ],
)

mediapipe_proto_library(
    name = "concatenate_vector_calculator_proto",
    srcs = ["concatenate_vector_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "dequantize_byte_array_calculator_proto",
    srcs = ["dequantize_byte_array_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "packet_cloner_calculator_proto",
    srcs = ["packet_cloner_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "packet_resampler_calculator_proto",
    srcs = ["packet_resampler_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "packet_thinner_calculator_proto",
    srcs = ["packet_thinner_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "split_vector_calculator_proto",
    srcs = ["split_vector_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "quantize_float_vector_calculator_proto",
    srcs = ["quantize_float_vector_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "sequence_shift_calculator_proto",
    srcs = ["sequence_shift_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "gate_calculator_proto",
    srcs = ["gate_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "constant_side_packet_calculator_proto",
    srcs = ["constant_side_packet_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/framework/formats:classification_proto",
        "//mediapipe/framework/formats:landmark_proto",
        "//mediapipe/framework/formats:matrix_data_proto",
        "//mediapipe/framework/formats:time_series_header_proto",
    ],
)

mediapipe_proto_library(
    name = "clip_vector_size_calculator_proto",
    srcs = ["clip_vector_size_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "flow_limiter_calculator_proto",
    srcs = ["flow_limiter_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "graph_profile_calculator_proto",
    srcs = ["graph_profile_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "get_vector_item_calculator_proto",
    srcs = ["get_vector_item_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "add_header_calculator",
    srcs = ["add_header_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_test(
    name = "add_header_calculator_test",
    size = "small",
    srcs = ["add_header_calculator_test.cc"],
    deps = [
        ":add_header_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:validate_type",
    ],
)

cc_library(
    name = "begin_loop_calculator",
    srcs = ["begin_loop_calculator.cc"],
    hdrs = ["begin_loop_calculator.h"],
    deps = [
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_contract",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:packet",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/gpu:gpu_buffer",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "end_loop_calculator",
    srcs = ["end_loop_calculator.cc"],
    hdrs = ["end_loop_calculator.h"],
    deps = [
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_contract",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/util:render_data_cc_proto",
        "@com_google_absl//absl/status",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
    alwayslink = 1,
)

cc_test(
    name = "begin_end_loop_calculator_graph_test",
    srcs = ["begin_end_loop_calculator_graph_test.cc"],
    deps = [
        ":begin_loop_calculator",
        ":end_loop_calculator",
        ":gate_calculator",
        ":pass_through_calculator",
        "//mediapipe/framework:calculator_contract",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:packet",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "begin_item_loop_calculator",
    srcs = ["begin_item_loop_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_contract",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:contract",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
    alwayslink = True,
)

cc_library(
    name = "end_item_loop_calculator",
    srcs = ["end_item_loop_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:contract",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/status",
    ],
    alwayslink = True,
)

cc_test(
    name = "begin_end_item_loop_calculator_graph_test",
    srcs = ["begin_end_item_loop_calculator_graph_test.cc"],
    args = ["--gunit_death_test_style=threadsafe"],  # The Docker test times out without this.
    deps = [
        ":begin_item_loop_calculator",
        ":end_item_loop_calculator",
        "//mediapipe/framework:calculator_contract",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:packet",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:status_matchers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "concatenate_vector_calculator_hdr",
    hdrs = ["concatenate_vector_calculator.h"],
    deps = [
        ":concatenate_vector_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "concatenate_vector_calculator",
    srcs = ["concatenate_vector_calculator.cc"],
    hdrs = ["concatenate_vector_calculator.h"],
    copts = select({
        # Needed for "//mediapipe/framework/formats:tensor" compatibility on Apple
        # platforms for Metal pulled in via the tensor.h header.
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":concatenate_vector_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/util:render_data_cc_proto",
        "@org_tensorflow//tensorflow/lite:framework",
    ] + select({
        ":apple_or_disable_gpu": [],
        "//conditions:default": [
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_buffer",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "concatenate_detection_vector_calculator",
    srcs = ["concatenate_detection_vector_calculator.cc"],
    deps = [
        ":concatenate_vector_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
    ],
    alwayslink = 1,
)

cc_library(
    name = "concatenate_proto_list_calculator",
    srcs = ["concatenate_proto_list_calculator.cc"],
    deps = [
        ":concatenate_vector_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:body_rig_cc_proto",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_test(
    name = "concatenate_proto_list_calculator_test",
    srcs = ["concatenate_proto_list_calculator_test.cc"],
    deps = [
        ":concatenate_proto_list_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "concatenate_vector_calculator_test",
    srcs = ["concatenate_vector_calculator_test.cc"],
    deps = [
        ":concatenate_vector_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "clip_vector_size_calculator",
    srcs = ["clip_vector_size_calculator.cc"],
    hdrs = ["clip_vector_size_calculator.h"],
    deps = [
        ":clip_vector_size_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
    alwayslink = 1,
)

cc_test(
    name = "clip_vector_size_calculator_test",
    srcs = ["clip_vector_size_calculator_test.cc"],
    deps = [
        ":clip_vector_size_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "counting_source_calculator",
    srcs = ["counting_source_calculator.cc"],
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
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_test(
    name = "make_pair_calculator_test",
    size = "small",
    srcs = ["make_pair_calculator_test.cc"],
    deps = [
        ":make_pair_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:validate_type",
        "//mediapipe/util:packet_test_util",
        "//mediapipe/util:time_series_test_util",
    ],
)

cc_library(
    name = "matrix_multiply_calculator",
    srcs = ["matrix_multiply_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/port:status",
        "@eigen_archive//:eigen3",
    ],
    alwayslink = 1,
)

cc_library(
    name = "matrix_subtract_calculator",
    srcs = ["matrix_subtract_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/port:status",
        "@eigen_archive//:eigen3",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mux_calculator",
    srcs = ["mux_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/stream_handler:mux_input_stream_handler",
    ],
    alwayslink = 1,
)

cc_library(
    name = "non_zero_calculator",
    srcs = ["non_zero_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/port:ret_check",
    ],
    alwayslink = 1,
)

cc_test(
    name = "non_zero_calculator_test",
    size = "small",
    srcs = ["non_zero_calculator_test.cc"],
    deps = [
        ":non_zero_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:validate_type",
    ],
)

cc_test(
    name = "mux_calculator_test",
    srcs = ["mux_calculator_test.cc"],
    deps = [
        ":mux_calculator",
        ":round_robin_demux_calculator",
        ":split_vector_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "packet_cloner_calculator",
    srcs = ["packet_cloner_calculator.cc"],
    deps = [
        ":packet_cloner_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_test(
    name = "packet_cloner_calculator_test",
    srcs = ["packet_cloner_calculator_test.cc"],
    deps = [
        ":gate_calculator",
        ":packet_cloner_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        "//mediapipe/framework/tool:sink",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "packet_inner_join_calculator",
    srcs = ["packet_inner_join_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_test(
    name = "packet_inner_join_calculator_test",
    srcs = ["packet_inner_join_calculator_test.cc"],
    deps = [
        ":packet_inner_join_calculator",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:validate_type",
    ],
)

cc_library(
    name = "packet_thinner_calculator",
    srcs = ["packet_thinner_calculator.cc"],
    deps = [
        ":packet_thinner_calculator_cc_proto",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:options_util",
        "@com_google_absl//absl/log:absl_check",
    ],
    alwayslink = 1,
)

cc_test(
    name = "packet_thinner_calculator_test",
    srcs = ["packet_thinner_calculator_test.cc"],
    deps = [
        ":packet_thinner_calculator",
        ":packet_thinner_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "pass_through_calculator",
    srcs = ["pass_through_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "round_robin_demux_calculator",
    srcs = ["round_robin_demux_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/port:ret_check",
    ],
    alwayslink = 1,
)

cc_library(
    name = "immediate_mux_calculator",
    srcs = ["immediate_mux_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/log:absl_log",
    ],
    alwayslink = 1,
)

cc_library(
    name = "packet_presence_calculator",
    srcs = ["packet_presence_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_test(
    name = "packet_presence_calculator_test",
    srcs = ["packet_presence_calculator_test.cc"],
    deps = [
        ":gate_calculator",
        ":packet_presence_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:sink",
    ],
)

cc_library(
    name = "previous_loopback_calculator",
    srcs = ["previous_loopback_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/stream_handler:immediate_input_stream_handler",
    ],
    alwayslink = 1,
)

cc_library(
    name = "flow_limiter_calculator",
    srcs = ["flow_limiter_calculator.cc"],
    deps = [
        ":flow_limiter_calculator_cc_proto",
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

cc_library(
    name = "string_to_int_calculator",
    srcs = ["string_to_int_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "default_side_packet_calculator",
    srcs = ["default_side_packet_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "side_packet_to_stream_calculator",
    srcs = ["side_packet_to_stream_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

cc_test(
    name = "side_packet_to_stream_calculator_test",
    srcs = ["side_packet_to_stream_calculator_test.cc"],
    deps = [
        ":side_packet_to_stream_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:options_util",
        "//mediapipe/util:packet_test_util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
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
    hdrs = ["packet_resampler_calculator.h"],
    deps = [
        ":packet_resampler_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:collection_item_id",
        "//mediapipe/framework/deps:mathutil",
        "//mediapipe/framework/deps:random",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:options_util",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_test(
    name = "packet_resampler_calculator_test",
    timeout = "short",
    srcs = [
        "packet_resampler_calculator_test.cc",
    ],
    deps = [
        ":packet_resampler_calculator",
        ":packet_resampler_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "previous_loopback_calculator_test",
    srcs = ["previous_loopback_calculator_test.cc"],
    deps = [
        ":gate_calculator",
        ":make_pair_calculator",
        ":pass_through_calculator",
        ":previous_loopback_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        "//mediapipe/framework/tool:sink",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "matrix_multiply_calculator_test",
    srcs = ["matrix_multiply_calculator_test.cc"],
    deps = [
        ":matrix_multiply_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/tool:validate_type",
        "@com_google_absl//absl/log:absl_check",
        "@eigen_archive//:eigen3",
    ],
)

cc_test(
    name = "matrix_subtract_calculator_test",
    srcs = ["matrix_subtract_calculator_test.cc"],
    deps = [
        ":matrix_subtract_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/tool:validate_type",
        "@eigen_archive//:eigen3",
    ],
)

cc_test(
    name = "flow_limiter_calculator_test",
    srcs = ["flow_limiter_calculator_test.cc"],
    deps = [
        ":counting_source_calculator",
        ":flow_limiter_calculator",
        ":flow_limiter_calculator_cc_proto",
        ":pass_through_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:test_calculators",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        "//mediapipe/framework/tool:simulation_clock",
        "//mediapipe/framework/tool:simulation_clock_executor",
        "//mediapipe/framework/tool:sink",
        "//mediapipe/util:packet_test_util",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "split_vector_calculator",
    srcs = ["split_vector_calculator.cc"],
    hdrs = ["split_vector_calculator.h"],
    copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":split_vector_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/util:resource_util",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ] + select({
        ":apple_or_disable_gpu": [],
        "//conditions:default": [
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_buffer",
        ],
    }),
    alwayslink = 1,
)

cc_test(
    name = "split_vector_calculator_test",
    srcs = ["split_vector_calculator_test.cc"],
    deps = [
        ":split_vector_calculator",
        ":split_vector_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/tool:validate_type",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "split_proto_list_calculator",
    srcs = ["split_proto_list_calculator.cc"],
    deps = [
        ":split_vector_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:body_rig_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/util:resource_util",
    ],
    alwayslink = 1,
)

cc_test(
    name = "split_proto_list_calculator_test",
    srcs = ["split_proto_list_calculator_test.cc"],
    deps = [
        ":split_proto_list_calculator",
        ":split_vector_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/tool:validate_type",
    ],
)

cc_library(
    name = "dequantize_byte_array_calculator",
    srcs = ["dequantize_byte_array_calculator.cc"],
    deps = [
        ":dequantize_byte_array_calculator_cc_proto",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_test(
    name = "dequantize_byte_array_calculator_test",
    srcs = ["dequantize_byte_array_calculator_test.cc"],
    deps = [
        ":dequantize_byte_array_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
    ],
)

cc_library(
    name = "quantize_float_vector_calculator",
    srcs = ["quantize_float_vector_calculator.cc"],
    deps = [
        ":quantize_float_vector_calculator_cc_proto",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_test(
    name = "quantize_float_vector_calculator_test",
    srcs = ["quantize_float_vector_calculator_test.cc"],
    deps = [
        ":quantize_float_vector_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
    ],
)

cc_library(
    name = "sequence_shift_calculator",
    srcs = ["sequence_shift_calculator.cc"],
    deps = [
        ":sequence_shift_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/log:absl_log",
    ],
    alwayslink = 1,
)

cc_test(
    name = "sequence_shift_calculator_test",
    srcs = ["sequence_shift_calculator_test.cc"],
    deps = [
        ":sequence_shift_calculator",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:validate_type",
    ],
)

cc_library(
    name = "gate_calculator",
    srcs = ["gate_calculator.cc"],
    deps = [
        ":gate_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/util:header_util",
    ],
    alwayslink = 1,
)

cc_test(
    name = "gate_calculator_test",
    srcs = ["gate_calculator_test.cc"],
    deps = [
        ":gate_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/log:absl_log",
    ],
)

cc_library(
    name = "matrix_to_vector_calculator",
    srcs = ["matrix_to_vector_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/tool:status_util",
        "//mediapipe/util:time_series_util",
        "@com_google_absl//absl/memory",
        "@eigen_archive//:eigen3",
    ],
    alwayslink = 1,
)

cc_test(
    name = "matrix_to_vector_calculator_test",
    srcs = ["matrix_to_vector_calculator_test.cc"],
    deps = [
        ":matrix_to_vector_calculator",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:validate_type",
        "//mediapipe/util:time_series_test_util",
        "//mediapipe/util:time_series_util",
    ],
)

cc_library(
    name = "merge_calculator",
    srcs = ["merge_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/log:absl_log",
    ],
    alwayslink = 1,
)

cc_test(
    name = "merge_calculator_test",
    srcs = ["merge_calculator_test.cc"],
    deps = [
        ":merge_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
    ],
)

cc_library(
    name = "stream_to_side_packet_calculator",
    srcs = ["stream_to_side_packet_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_test(
    name = "stream_to_side_packet_calculator_test",
    srcs = ["stream_to_side_packet_calculator_test.cc"],
    deps = [
        ":stream_to_side_packet_calculator",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "constant_side_packet_calculator",
    srcs = ["constant_side_packet_calculator.cc"],
    deps = [
        ":constant_side_packet_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:collection_item_id",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:matrix_data_cc_proto",
        "//mediapipe/framework/formats:time_series_header_cc_proto",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_test(
    name = "constant_side_packet_calculator_test",
    srcs = ["constant_side_packet_calculator_test.cc"],
    deps = [
        ":constant_side_packet_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "graph_profile_calculator",
    srcs = ["graph_profile_calculator.cc"],
    deps = [
        ":graph_profile_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_profile_cc_proto",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_test(
    name = "graph_profile_calculator_test",
    srcs = ["graph_profile_calculator_test.cc"],
    deps = [
        ":graph_profile_calculator",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_profile_cc_proto",
        "//mediapipe/framework:test_calculators",
        "//mediapipe/framework/deps:clock",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:threadpool",
        "//mediapipe/framework/tool:simulation_clock_executor",
        "//mediapipe/framework/tool:sink",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "get_vector_item_calculator",
    srcs = ["get_vector_item_calculator.cc"],
    hdrs = ["get_vector_item_calculator.h"],
    deps = [
        ":get_vector_item_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_test(
    name = "get_vector_item_calculator_test",
    srcs = ["get_vector_item_calculator_test.cc"],
    deps = [
        ":get_vector_item_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "vector_indices_calculator",
    srcs = ["vector_indices_calculator.cc"],
    hdrs = ["vector_indices_calculator.h"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_test(
    name = "vector_indices_calculator_test",
    srcs = ["vector_indices_calculator_test.cc"],
    deps = [
        ":vector_indices_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
    ],
)

cc_library(
    name = "vector_size_calculator",
    srcs = ["vector_size_calculator.cc"],
    hdrs = ["vector_size_calculator.h"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "packet_sequencer_calculator",
    srcs = ["packet_sequencer_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:contract",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/stream_handler:immediate_input_stream_handler",
    ],
    alwayslink = 1,
)

cc_test(
    name = "packet_sequencer_calculator_test",
    srcs = ["packet_sequencer_calculator_test.cc"],
    deps = [
        ":packet_sequencer_calculator",
        ":pass_through_calculator",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:subgraph",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "merge_to_vector_calculator",
    srcs = ["merge_to_vector_calculator.cc"],
    hdrs = ["merge_to_vector_calculator.h"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:packet",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "bypass_calculator_proto",
    srcs = ["bypass_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "bypass_calculator",
    srcs = ["bypass_calculator.cc"],
    deps = [
        ":bypass_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:collection_item_id",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_test(
    name = "bypass_calculator_test",
    srcs = ["bypass_calculator_test.cc"],
    deps = [
        ":bypass_calculator",
        ":pass_through_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:switch_container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "value_or_default_calculator",
    srcs = ["value_or_default_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = True,
)

cc_test(
    name = "value_or_default_calculator_test",
    srcs = ["value_or_default_calculator_test.cc"],
    deps = [
        ":value_or_default_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:packet",
        "//mediapipe/framework/port:gtest_main",
    ],
)
