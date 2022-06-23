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

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")
load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "data_as_c_string",
    "mediapipe_binary_graph",
)
load("//mediapipe/framework:mediapipe_cc_test.bzl", "mediapipe_cc_test")
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

licenses(["notice"])

package(default_visibility = ["//visibility:private"])

exports_files([
    "simple_subgraph_template.cc",
    "options_lib_template.cc",
])

bzl_library(
    name = "mediapipe_graph_bzl",
    srcs = [
        "mediapipe_graph.bzl",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":build_defs_bzl",
        "//mediapipe/framework:encode_binary_proto",
        "//mediapipe/framework:transitive_protos_bzl",
        "//mediapipe/framework/deps:descriptor_set_bzl",
        "//mediapipe/framework/deps:expand_template_bzl",
        "@org_tensorflow//tensorflow/lite/core/shims:cc_library_with_tflite_bzl",
    ],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = [
        "build_defs.bzl",
    ],
    visibility = [
        "//mediapipe/app/xeno/catalog:__subpackages__",
        "//mediapipe/framework:__subpackages__",
    ],
)

cc_library(
    name = "text_to_binary_graph",
    srcs = ["text_to_binary_graph.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/port:advanced_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

mediapipe_proto_library(
    name = "calculator_graph_template_proto",
    srcs = ["calculator_graph_template.proto"],
    def_options_lib = False,
    def_py_proto = False,
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/framework/deps:proto_descriptor_proto",
    ],
)

mediapipe_proto_library(
    name = "packet_generator_wrapper_calculator_proto",
    srcs = ["packet_generator_wrapper_calculator.proto"],
    def_py_proto = False,
    visibility = ["//mediapipe/framework:mediapipe_internal"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:packet_generator_proto",
    ],
)

mediapipe_proto_library(
    name = "source_proto",
    srcs = ["source.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
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
    visibility = ["//visibility:public"],
    deps = [
        ":validate_name",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/port:map_util",
        "@com_google_absl//absl/container:flat_hash_set",
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
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
    ],
)

cc_library(
    name = "options_map",
    hdrs = ["options_map.h"],
    visibility = ["//mediapipe/framework:mediapipe_internal"],
    deps = [
        ":type_util",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/port:any_proto",
        "//mediapipe/framework/port:status",
    ],
)

mediapipe_proto_library(
    name = "field_data_proto",
    srcs = ["field_data.proto"],
    visibility = ["//visibility:public"],
    deps = ["@com_google_protobuf//:any_proto"],
)

cc_library(
    name = "options_field_util",
    srcs = ["options_field_util.cc"],
    hdrs = ["options_field_util.h"],
    visibility = ["//mediapipe/framework:mediapipe_internal"],
    deps = [
        ":field_data_cc_proto",
        ":name_util",
        ":options_registry",
        ":proto_util_lite",
        ":type_util",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:packet_type",
        "//mediapipe/framework/port:advanced_proto",
        "//mediapipe/framework/port:any_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "options_syntax_util",
    srcs = ["options_syntax_util.cc"],
    hdrs = ["options_syntax_util.h"],
    visibility = ["//mediapipe/framework:mediapipe_internal"],
    deps = [
        ":name_util",
        ":options_field_util",
        ":options_registry",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:packet_type",
        "//mediapipe/framework/port:advanced_proto",
        "//mediapipe/framework/port:any_proto",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "options_util",
    srcs = ["options_util.cc"],
    hdrs = ["options_util.h"],
    visibility = ["//mediapipe/framework:mediapipe_internal"],
    deps = [
        ":options_field_util",
        ":options_map",
        ":options_registry",
        ":options_syntax_util",
        ":proto_util_lite",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:collection",
        "//mediapipe/framework:input_stream_shard",
        "//mediapipe/framework:output_side_packet",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:packet_set",
        "//mediapipe/framework:packet_type",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:advanced_proto",
        "//mediapipe/framework/port:any_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:name_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "message_type_util",
    srcs = ["message_type_util.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/port:advanced_proto",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:logging",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "options_registry",
    srcs = ["options_registry.cc"],
    hdrs = ["options_registry.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":field_data_cc_proto",
        ":proto_util_lite",
        "//mediapipe/framework/deps:registration",
        "//mediapipe/framework/port:advanced_proto",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
    ],
)

mediapipe_cc_test(
    name = "options_util_test",
    size = "small",
    timeout = "moderate",
    srcs = ["options_util_test.cc"],
    # A non-empty "data" param is needed to build the "_test_wasm" target.
    data = [":node_chain_subgraph.proto"],
    requires_full_emulation = False,
    deps = [
        ":options_field_util",
        ":options_registry",
        ":options_syntax_util",
        ":options_util",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator_cc_proto",
        "//mediapipe/framework:basic_types_registration",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:validated_graph_config",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/testdata:night_light_calculator_options_lib",
        "//mediapipe/framework/tool:node_chain_subgraph_options_lib",
        "//mediapipe/util:header_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "packet_generator_wrapper_calculator",
    srcs = ["packet_generator_wrapper_calculator.cc"],
    visibility = ["//mediapipe/framework:mediapipe_internal"],
    deps = [
        ":packet_generator_wrapper_calculator_cc_proto",
        "//mediapipe/framework:calculator_base",
        "//mediapipe/framework:calculator_registry",
        "//mediapipe/framework:output_side_packet",
        "//mediapipe/framework:packet_generator",
    ],
    alwayslink = 1,
)

cc_library(
    name = "proto_util_lite",
    srcs = ["proto_util_lite.cc"],
    hdrs = ["proto_util_lite.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":field_data_cc_proto",
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
        "//mediapipe/framework:graph_service_manager",
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
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tag_map_helper",
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
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:demangle",
        "//mediapipe/framework:port",
        "@com_google_absl//absl/base:core_headers",
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
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:legacy_calculator_support",
        "//mediapipe/framework:packet_generator",
        "//mediapipe/framework:packet_generator_cc_proto",
        "//mediapipe/framework:packet_set",
        "//mediapipe/framework:packet_type",
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

mediapipe_proto_library(
    name = "node_chain_subgraph_proto",
    srcs = ["node_chain_subgraph.proto"],
    visibility = ["//mediapipe:__subpackages__"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
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
        ":node_chain_subgraph_options_lib",
        ":subgraph_expansion",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:graph_service_manager",
        "//mediapipe/framework:mediapipe_options_cc_proto",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:packet_generator_cc_proto",
        "//mediapipe/framework:packet_set",
        "//mediapipe/framework:packet_type",
        "//mediapipe/framework:status_handler",
        "//mediapipe/framework:subgraph",
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
        "//mediapipe/framework/deps:no_destructor",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:advanced_proto",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@stblib//:stb_image",
        "@stblib//:stb_image_write",
    ],
)

cc_library(
    name = "container_util",
    srcs = ["container_util.cc"],
    hdrs = ["container_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":name_util",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:switch_container_cc_proto",
    ],
)

cc_library(
    name = "switch_demux_calculator",
    srcs = ["switch_demux_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":container_util",
        ":options_util",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:collection_item_id",
        "//mediapipe/framework/deps:mathutil",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        "//mediapipe/framework/tool:switch_container_cc_proto",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "switch_mux_calculator",
    srcs = ["switch_mux_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":container_util",
        ":options_util",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:collection_item_id",
        "//mediapipe/framework:input_stream_shard",
        "//mediapipe/framework:output_stream_shard",
        "//mediapipe/framework/deps:mathutil",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        "//mediapipe/framework/tool:switch_container_cc_proto",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "switch_container_proto",
    srcs = ["switch_container.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "switch_container",
    srcs = ["switch_container.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":container_util",
        ":name_util",
        ":subgraph_expansion",
        ":switch_demux_calculator",
        ":switch_mux_calculator",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:mediapipe_options_cc_proto",
        "//mediapipe/framework:subgraph",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:switch_container_cc_proto",
    ],
    alwayslink = 1,
)

cc_test(
    name = "switch_container_test",
    size = "small",
    srcs = ["switch_container_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":node_chain_subgraph_cc_proto",
        ":subgraph_expansion",
        ":switch_container",
        "//mediapipe/calculators/core:pass_through_calculator",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:subgraph",
        "//mediapipe/framework:test_calculators",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        "@com_google_absl//absl/strings",
    ],
)

exports_files(
    ["build_defs.bzl"],
    visibility = [
        "//mediapipe/framework:__subpackages__",
    ],
)
