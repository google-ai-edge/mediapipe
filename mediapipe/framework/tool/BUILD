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

load(
    "//mediapipe/framework/port:build_config.bzl",
    "mediapipe_cc_proto_library",
)
load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "data_as_c_string",
    "mediapipe_binary_graph",
)

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:private"])

exports_files([
    "simple_subgraph_template.cc",
])

cc_library(
    name = "text_to_binary_graph",
    srcs = ["text_to_binary_graph.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/port:advanced_proto",
        "//mediapipe/framework/port:commandlineflags",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
)

proto_library(
    name = "calculator_graph_template_proto",
    srcs = ["calculator_graph_template.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/framework/deps:proto_descriptor_proto",
    ],
)

java_proto_library(
    name = "calculator_graph_template_java_proto",
    visibility = ["//visibility:public"],
    deps = [":calculator_graph_template_proto"],
)

java_lite_proto_library(
    name = "calculator_graph_template_java_proto_lite",
    strict_deps = 0,
    visibility = ["//visibility:public"],
    deps = [":calculator_graph_template_proto"],
)

proto_library(
    name = "source_proto",
    srcs = ["source.proto"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework:calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "calculator_graph_template_cc_proto",
    srcs = ["calculator_graph_template.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/deps:proto_descriptor_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":calculator_graph_template_proto"],
)

mediapipe_cc_proto_library(
    name = "source_cc_proto",
    srcs = ["source.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":source_proto"],
)

cc_binary(
    name = "encode_as_c_string",
    srcs = ["encode_as_c_string.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "fill_packet_set",
    srcs = ["fill_packet_set.cc"],
    hdrs = ["fill_packet_set.h"],
    visibility = ["//mediapipe/framework:mediapipe_internal"],
    deps = [
        "//mediapipe/framework:packet_set",
        "//mediapipe/framework:packet_type",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "//mediapipe/framework/tool:status_util",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "name_util",
    srcs = ["name_util.cc"],
    hdrs = ["name_util.h"],
    visibility = ["//mediapipe/framework:mediapipe_internal"],
    deps = [
        ":validate_name",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/port:map_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "executor_util",
    srcs = ["executor_util.cc"],
    hdrs = ["executor_util.h"],
    visibility = ["//mediapipe/framework:mediapipe_internal"],
    deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:mediapipe_options_cc_proto",
        "//mediapipe/framework:thread_pool_executor_cc_proto",
        "//mediapipe/framework/port:integral_types",
    ],
)

cc_test(
    name = "executor_util_test",
    srcs = ["executor_util_test.cc"],
    visibility = ["//mediapipe/framework:mediapipe_internal"],
    deps = [
        ":executor_util",
        "//mediapipe/framework/deps:message_matchers",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
    ],
)

cc_library(
    name = "options_util",
    hdrs = ["options_util.h"],
    visibility = ["//mediapipe/framework:mediapipe_internal"],
    deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:packet_generator_cc_proto",
        "//mediapipe/framework:packet_set",
        "//mediapipe/framework/port:any_proto",
        "//mediapipe/framework/tool:type_util",
    ],
)

cc_library(
    name = "packet_util",
    hdrs = ["packet_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:packet",
        "//mediapipe/framework/port:statusor",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "proto_util_lite",
    srcs = ["proto_util_lite.cc"],
    hdrs = ["proto_util_lite.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:type_map",
        "//mediapipe/framework/port:advanced_proto_lite",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "simulation_clock",
    srcs = ["simulation_clock.cc"],
    hdrs = ["simulation_clock.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/deps:clock",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "sink",
    srcs = ["sink.cc"],
    hdrs = ["sink.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":name_util",
        "//mediapipe/calculators/internal:callback_packet_calculator",
        "//mediapipe/calculators/internal:callback_packet_calculator_cc_proto",
        "//mediapipe/framework:calculator_base",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_graph",
        "//mediapipe/framework:calculator_registry",
        "//mediapipe/framework:input_stream",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:packet_type",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:source_location",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = 1,
)

cc_library(
    name = "subgraph_expansion",
    srcs = ["subgraph_expansion.cc"],
    hdrs = ["subgraph_expansion.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":name_util",
        ":tag_map",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:packet_generator",
        "//mediapipe/framework:packet_generator_cc_proto",
        "//mediapipe/framework:port",
        "//mediapipe/framework:status_handler_cc_proto",
        "//mediapipe/framework:subgraph",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:map_util",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "source",
    srcs = ["source.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:source_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "status_util",
    srcs = ["status_util.cc"],
    hdrs = ["status_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tag_map",
    srcs = ["tag_map.cc"],
    hdrs = ["tag_map.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":validate_name",
        "//mediapipe/framework:collection_item_id",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tag_map_helper",
    testonly = 1,
    srcs = ["tag_map_helper.cc"],
    hdrs = ["tag_map_helper.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tag_map",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "template_expander",
    srcs = ["template_expander.cc"],
    hdrs = ["template_expander.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":proto_util_lite",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/deps:proto_descriptor_cc_proto",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:numbers",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:calculator_graph_template_cc_proto",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "template_parser",
    srcs = [
        "template_parser.cc",
    ],
    hdrs = [
        "template_parser.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":proto_util_lite",
        "//mediapipe/framework/deps:proto_descriptor_cc_proto",
        "//mediapipe/framework/port:advanced_proto",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:map_util",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:calculator_graph_template_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "type_util",
    hdrs = ["type_util.h"],
    visibility = ["//mediapipe/framework:mediapipe_internal"],
    deps = [
        "//mediapipe/framework:port",
    ],
)

cc_library(
    name = "validate",
    srcs = ["validate.cc"],
    hdrs = ["validate.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":validate_name",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
)

cc_library(
    name = "validate_name",
    srcs = ["validate_name.cc"],
    hdrs = ["validate_name.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:source_location",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "fill_packet_set_test",
    size = "small",
    srcs = ["fill_packet_set_test.cc"],
    deps = [
        ":fill_packet_set",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/port:gtest_main",
    ],
)

cc_test(
    name = "sink_test",
    size = "small",
    srcs = [
        "sink_test.cc",
    ],
    deps = [
        ":sink",
        ":validate_type",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "status_util_test",
    size = "small",
    srcs = ["status_util_test.cc"],
    deps = [
        ":status_util",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:status",
    ],
)

cc_test(
    name = "tag_map_test",
    size = "small",
    srcs = ["tag_map_test.cc"],
    deps = [
        ":tag_map",
        ":tag_map_helper",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:map_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "validate_name_test",
    size = "small",
    srcs = ["validate_name_test.cc"],
    linkstatic = 1,
    deps = [
        ":validate_name",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/deps:message_matchers",
        "//mediapipe/framework/port:gtest_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "validate_type",
    # TODO make this testonly.
    srcs = ["validate_type.cc"],
    hdrs = ["validate_type.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":status_util",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_contract",
        "//mediapipe/framework:legacy_calculator_support",
        "//mediapipe/framework:packet_generator",
        "//mediapipe/framework:packet_generator_cc_proto",
        "//mediapipe/framework:packet_set",
        "//mediapipe/framework:packet_type",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:map_util",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:threadpool",
        "//mediapipe/util:cpu_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "simulation_clock_executor",
    srcs = ["simulation_clock_executor.cc"],
    hdrs = ["simulation_clock_executor.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:thread_pool_executor",
        "//mediapipe/framework/tool:simulation_clock",
    ],
)

cc_test(
    name = "simulation_clock_test",
    srcs = ["simulation_clock_test.cc"],
    deps = [
        ":simulation_clock",
        ":simulation_clock_executor",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:immediate_mux_calculator",
        "//mediapipe/calculators/core:round_robin_demux_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:executor",
        "//mediapipe/framework:input_stream",
        "//mediapipe/framework:output_stream",
        "//mediapipe/framework:test_calculators",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

mediapipe_binary_graph(
    name = "test_binarypb",
    graph = "//mediapipe/framework/tool/testdata:test_graph",
    output_name = "test.binarypb",
    visibility = ["//visibility:private"],
)

data_as_c_string(
    name = "test_binarypb_inc",
    testonly = 1,
    srcs = [":test_binarypb"],
    outs = ["test_binarypb.inc"],
)

proto_library(
    name = "node_chain_subgraph_proto",
    srcs = ["node_chain_subgraph.proto"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "node_chain_subgraph_cc_proto",
    srcs = ["node_chain_subgraph.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//mediapipe:__subpackages__"],
    deps = [":node_chain_subgraph_proto"],
)

cc_test(
    name = "data_as_c_string_test",
    srcs = [
        "data_as_c_string_test.cc",
        ":test_binarypb_inc",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:gtest_main",
    ],
)

cc_test(
    name = "subgraph_expansion_test",
    size = "small",
    srcs = ["subgraph_expansion_test.cc"],
    deps = [
        ":node_chain_subgraph_cc_proto",
        ":subgraph_expansion",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:mediapipe_options_cc_proto",
        "//mediapipe/framework:packet_generator_cc_proto",
        "//mediapipe/framework:packet_set",
        "//mediapipe/framework:packet_type",
        "//mediapipe/framework:status_handler",
        "//mediapipe/framework:subgraph",
        "//mediapipe/framework/deps:message_matchers",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/tool/testdata:dub_quad_test_subgraph",
        "//mediapipe/framework/tool/testdata:nested_test_subgraph",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:advanced_proto",
        "//mediapipe/framework/port:gtest",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//mediapipe/framework/deps:no_destructor",
        "//mediapipe/framework/port:logging",
    ] + select({
        "//conditions:default": [
            "//mediapipe/framework/port:file_helpers",
        ],
        "//mediapipe:android": [
            "//mediapipe/util/android/file/base",
        ],
        "//mediapipe:ios": [],
        "//mediapipe:macos": [
            "//mediapipe/framework/port:file_helpers",
        ],
    }),
)

exports_files(
    ["build_defs.bzl"],
    visibility = ["//mediapipe/framework:__subpackages__"],
)
