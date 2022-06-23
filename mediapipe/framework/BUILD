#
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
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

licenses(["notice"])

package(default_visibility = ["//visibility:private"])

package_group(
    name = "mediapipe_internal",
    packages = [
        "//mediapipe/...",
    ],
)

bzl_library(
    name = "transitive_protos_bzl",
    srcs = [
        "transitive_protos.bzl",
    ],
    visibility = ["//mediapipe/framework:__subpackages__"],
)

bzl_library(
    name = "encode_binary_proto_bzl",
    srcs = [
        "encode_binary_proto.bzl",
    ],
    visibility = ["//visibility:public"],
)

alias(
    name = "encode_binary_proto",
    actual = ":encode_binary_proto_bzl",
    deprecation = "Use encode_binary_proto_bzl",
    visibility = ["//visibility:public"],
)

mediapipe_proto_library(
    name = "calculator_proto",
    srcs = ["calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:mediapipe_options_proto",
        "//mediapipe/framework:packet_factory_proto",
        "//mediapipe/framework:packet_generator_proto",
        "//mediapipe/framework:status_handler_proto",
        "//mediapipe/framework:stream_handler_proto",
        "@com_google_protobuf//:any_proto",
    ],
)

mediapipe_proto_library(
    name = "calculator_options_proto",
    srcs = ["calculator_options.proto"],
    visibility = ["//visibility:public"],
)

mediapipe_proto_library(
    name = "calculator_contract_test_proto",
    testonly = 1,
    srcs = ["calculator_contract_test.proto"],
    visibility = ["//mediapipe/framework:__subpackages__"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "calculator_profile_proto",
    srcs = ["calculator_profile.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "mediapipe_options_proto",
    srcs = ["mediapipe_options.proto"],
    visibility = [":mediapipe_internal"],
)

mediapipe_proto_library(
    name = "packet_factory_proto",
    srcs = ["packet_factory.proto"],
    visibility = [":mediapipe_internal"],
)

mediapipe_proto_library(
    name = "packet_generator_proto",
    srcs = ["packet_generator.proto"],
    visibility = [":mediapipe_internal"],
)

mediapipe_proto_library(
    name = "packet_test_proto",
    testonly = 1,
    srcs = ["packet_test.proto"],
    visibility = [
        ":mediapipe_internal",
        "//mediapipe/framework:__subpackages__",
    ],
)

mediapipe_proto_library(
    name = "status_handler_proto",
    srcs = ["status_handler.proto"],
    visibility = [":mediapipe_internal"],
    deps = ["//mediapipe/framework:mediapipe_options_proto"],
)

mediapipe_proto_library(
    name = "stream_handler_proto",
    srcs = ["stream_handler.proto"],
    visibility = [":mediapipe_internal"],
    deps = ["//mediapipe/framework:mediapipe_options_proto"],
)

mediapipe_proto_library(
    name = "test_calculators_proto",
    testonly = 1,
    srcs = ["test_calculators.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "thread_pool_executor_proto",
    srcs = ["thread_pool_executor.proto"],
    visibility = [":mediapipe_internal"],
    deps = ["//mediapipe/framework:mediapipe_options_proto"],
)

config_setting(
    name = "android_no_jni",
    define_values = {"MEDIAPIPE_NO_JNI": "1"},
    values = {"crosstool_top": "//external:android/crosstool"},
    visibility = ["//visibility:public"],
)

cc_library(
    name = "calculator_base",
    srcs = ["calculator_base.cc"],
    hdrs = ["calculator_base.h"],
    visibility = [
        ":mediapipe_internal",
    ],
    deps = [
        ":calculator_context",
        ":calculator_contract",
        ":port",
        ":timestamp",
        "//mediapipe/framework/deps:registration",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "calculator_context",
    srcs = ["calculator_context.cc"],
    hdrs = ["calculator_context.h"],
    visibility = [":mediapipe_internal"],
    deps = [
        ":calculator_state",
        ":counter",
        ":graph_service",
        ":input_stream_shard",
        ":output_stream_shard",
        ":packet",
        ":packet_set",
        ":port",
        ":timestamp",
        "//mediapipe/framework/port:any_proto",
        "//mediapipe/framework/port:status",
    ],
)

cc_library(
    name = "calculator_context_manager",
    srcs = ["calculator_context_manager.cc"],
    hdrs = ["calculator_context_manager.h"],
    visibility = [":mediapipe_internal"],
    deps = [
        ":calculator_context",
        ":calculator_state",
        ":timestamp",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:tag_map",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "calculator_contract",
    srcs = ["calculator_contract.cc"],
    hdrs = ["calculator_contract.h"],
    visibility = [
        ":mediapipe_internal",
    ],
    deps = [
        ":graph_service",
        ":packet_type",
        ":port",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:mediapipe_options_cc_proto",
        "//mediapipe/framework:packet_generator_cc_proto",
        "//mediapipe/framework:status_handler_cc_proto",
        "//mediapipe/framework/port:any_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:options_map",
        "//mediapipe/framework/tool:packet_generator_wrapper_calculator_cc_proto",
        "//mediapipe/framework/tool:tag_map",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "legacy_calculator_support",
    srcs = ["legacy_calculator_support.cc"],
    hdrs = ["legacy_calculator_support.h"],
    visibility = [
        ":mediapipe_internal",
    ],
    deps = [
        ":calculator_context",
        ":calculator_contract",
    ],
)

cc_library(
    name = "calculator_framework",
    hdrs = ["calculator_framework.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":calculator_base",
        ":calculator_graph",
        ":calculator_registry",
        ":counter_factory",
        ":input_stream",
        ":output_side_packet",
        ":output_stream",
        ":packet",
        ":packet_generator",
        ":packet_generator_graph",
        ":packet_set",
        ":packet_type",
        ":port",
        ":status_handler",
        ":subgraph",
        ":timestamp",
        ":validated_graph_config",
        "//mediapipe/framework/tool:sink",
        "//mediapipe/framework/tool:status_util",
        "//mediapipe/framework/tool:validate",
        "//mediapipe/framework/tool:validate_name",
    ],
)

cc_library(
    name = "calculator_graph",
    srcs = [
        "calculator_graph.cc",
        "scheduler.cc",
    ],
    hdrs = [
        "calculator_graph.h",
        "scheduler.h",
    ],
    visibility = [
        ":mediapipe_internal",
    ],
    deps = [
        ":calculator_base",
        ":calculator_node",
        ":counter_factory",
        ":delegating_executor",
        ":mediapipe_profiling",
        ":executor",
        ":graph_output_stream",
        ":graph_service",
        ":graph_service_manager",
        ":input_stream_manager",
        ":output_side_packet_impl",
        ":output_stream",
        ":output_stream_manager",
        ":output_stream_poller",
        ":output_stream_shard",
        ":packet",
        ":packet_generator",
        ":packet_generator_graph",
        ":packet_set",
        ":packet_type",
        ":port",
        ":scheduler_queue",
        ":status_handler",
        ":thread_pool_executor",
        ":timestamp",
        ":validated_graph_config",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:packet_generator_cc_proto",
        "//mediapipe/framework:status_handler_cc_proto",
        "//mediapipe/framework:thread_pool_executor_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:source_location",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:fill_packet_set",
        "//mediapipe/framework/tool:packet_generator_wrapper_calculator",
        "//mediapipe/framework/tool:status_util",
        "//mediapipe/framework/tool:tag_map",
        "//mediapipe/framework/tool:validate",
        "//mediapipe/framework/tool:validate_name",
        "//mediapipe/gpu:graph_support",
        "//mediapipe/gpu:gpu_service",
        "//mediapipe/util:cpu_util",
    ] + select({
        "//conditions:default": ["//mediapipe/gpu:gpu_shared_data_internal"],
        "//mediapipe/gpu:disable_gpu": [],
    }),
)

cc_library(
    name = "graph_service_manager",
    srcs = ["graph_service_manager.cc"],
    hdrs = ["graph_service_manager.h"],
    visibility = [":mediapipe_internal"],
    deps = [
        ":graph_service",
        "//mediapipe/framework:packet",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "graph_service_manager_test",
    srcs = ["graph_service_manager_test.cc"],
    deps = [
        ":graph_service_manager",
        "//mediapipe/framework:packet",
        "//mediapipe/framework/port:gtest_main",
    ],
)

cc_library(
    name = "calculator_node",
    srcs = ["calculator_node.cc"],
    hdrs = ["calculator_node.h"],
    visibility = [":mediapipe_internal"],
    deps = [
        ":calculator_base",
        ":calculator_context",
        ":calculator_context_manager",
        ":calculator_state",
        ":counter_factory",
        ":input_side_packet_handler",
        ":input_stream_handler",
        ":input_stream_manager",
        ":legacy_calculator_support",
        ":mediapipe_profiling",
        ":output_side_packet_impl",
        ":output_stream_handler",
        ":output_stream_manager",
        ":packet",
        ":packet_set",
        ":packet_type",
        ":port",
        ":timestamp",
        ":validated_graph_config",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:stream_handler_cc_proto",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:source_location",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/stream_handler:default_input_stream_handler",
        "//mediapipe/framework/stream_handler:in_order_output_stream_handler",
        "//mediapipe/framework/tool:name_util",
        "//mediapipe/framework/tool:status_util",
        "//mediapipe/framework/tool:tag_map",
        "//mediapipe/framework/tool:validate_name",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "calculator_registry",
    hdrs = ["calculator_registry.h"],
    visibility = [
        ":mediapipe_internal",
    ],
    deps = [
        ":calculator_base",
    ],
)

cc_library(
    name = "calculator_runner",
    testonly = 1,
    srcs = ["calculator_runner.cc"],
    hdrs = ["calculator_runner.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":calculator_framework",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:sink",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "calculator_state",
    srcs = ["calculator_state.cc"],
    hdrs = ["calculator_state.h"],
    visibility = [":mediapipe_internal"],
    deps = [
        ":counter",
        ":counter_factory",
        ":graph_service",
        ":graph_service_manager",
        ":input_stream",
        ":output_stream",
        ":packet",
        ":packet_set",
        ":port",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/port:any_proto",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/tool:options_map",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "camera_intrinsics",
    hdrs = ["camera_intrinsics.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "collection",
    hdrs = ["collection.h"],
    visibility = [":mediapipe_internal"],
    deps = [
        ":collection_item_id",
        ":type_map",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/tool:tag_map",
        "//mediapipe/framework/tool:tag_map_helper",
        "//mediapipe/framework/tool:validate_name",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "collection_item_id",
    srcs = ["collection_item_id.cc"],
    hdrs = ["collection_item_id.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/deps:intops",
    ],
)

cc_library(
    name = "counter",
    hdrs = ["counter.h"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework/port:integral_types"],
)

cc_library(
    name = "counter_factory",
    srcs = ["counter_factory.cc"],
    hdrs = ["counter_factory.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":counter",
        ":port",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:map_util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "delegating_executor",
    srcs = ["delegating_executor.cc"],
    hdrs = ["delegating_executor.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":executor",
    ],
)

cc_library(
    name = "demangle",
    hdrs = ["demangle.h"],
    defines = select({
        "//mediapipe/framework/profiler:android_release": [
            "MEDIAPIPE_HAS_CXA_DEMANGLE=0",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mediapipe_profiling",
    hdrs = [
        "mediapipe_profiling.h",
        "platform_specific_profiling.h",
    ],
    visibility = [
        ":mediapipe_internal",
    ],
    deps = [
        "//mediapipe/framework/profiler:graph_profiler",
    ],
)

cc_library(
    name = "executor",
    srcs = ["executor.cc"],
    hdrs = ["executor.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:mediapipe_options_cc_proto",
        "//mediapipe/framework/deps:registration",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
    ],
)

cc_library(
    name = "graph_output_stream",
    srcs = ["graph_output_stream.cc"],
    hdrs = ["graph_output_stream.h"],
    visibility = [
        ":mediapipe_internal",
    ],
    deps = [
        ":input_stream_handler",
        ":input_stream_manager",
        ":output_stream_manager",
        ":packet",
        ":packet_set",
        ":packet_type",
        ":timestamp",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "graph_service",
    hdrs = ["graph_service.h"],
    visibility = [":mediapipe_internal"],
    deps = [
        ":packet",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "input_side_packet_handler",
    srcs = ["input_side_packet_handler.cc"],
    hdrs = ["input_side_packet_handler.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":collection_item_id",
        ":packet",
        ":packet_set",
        ":packet_type",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:fill_packet_set",
    ],
)

cc_library(
    name = "input_stream",
    hdrs = ["input_stream.h"],
    visibility = [":mediapipe_internal"],
    deps = [
        ":packet",
        ":port",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "input_stream_handler",
    srcs = ["input_stream_handler.cc"],
    hdrs = ["input_stream_handler.h"],
    visibility = [
        ":mediapipe_internal",
        "//research/interaction/mediapipe/calculators:__pkg__",
    ],
    deps = [
        ":calculator_context",
        ":calculator_context_manager",
        ":collection",
        ":collection_item_id",
        ":input_stream_manager",
        ":input_stream_shard",
        ":mediapipe_profiling",
        ":packet",
        ":packet_set",
        ":packet_type",
        "//mediapipe/framework:mediapipe_options_cc_proto",
        "//mediapipe/framework/deps:registration",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:tag_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "input_stream_manager",
    srcs = ["input_stream_manager.cc"],
    hdrs = ["input_stream_manager.h"],
    visibility = [":mediapipe_internal"],
    deps = [
        ":packet",
        ":packet_type",
        ":port",
        ":timestamp",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:source_location",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:status_util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "input_stream_shard",
    srcs = ["input_stream_shard.cc"],
    hdrs = ["input_stream_shard.h"],
    visibility = [":mediapipe_internal"],
    deps = [
        ":input_stream",
        ":packet",
        ":packet_type",
        ":port",
        ":timestamp",
        "//mediapipe/framework/port:source_location",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:status_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "lifetime_tracker",
    testonly = 1,
    hdrs = ["lifetime_tracker.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "output_side_packet",
    hdrs = ["output_side_packet.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":packet",
    ],
)

cc_library(
    name = "output_side_packet_impl",
    srcs = ["output_side_packet_impl.cc"],
    hdrs = ["output_side_packet_impl.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":collection_item_id",
        ":input_side_packet_handler",
        ":output_side_packet",
        ":packet",
        ":packet_type",
        ":port",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:source_location",
        "//mediapipe/framework/port:status",
    ],
)

cc_library(
    name = "output_stream",
    hdrs = ["output_stream.h"],
    visibility = [":mediapipe_internal"],
    deps = [
        ":packet",
        ":port",
        ":timestamp",
        "//mediapipe/framework/port:logging",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "output_stream_handler",
    srcs = ["output_stream_handler.cc"],
    hdrs = ["output_stream_handler.h"],
    visibility = [
        ":mediapipe_internal",
    ],
    deps = [
        ":calculator_context_manager",
        ":collection",
        ":collection_item_id",
        ":output_stream_manager",
        ":output_stream_shard",
        ":packet_set",
        ":packet_type",
        ":timestamp",
        "//mediapipe/framework:mediapipe_options_cc_proto",
        "//mediapipe/framework/deps:registration",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:tag_map",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "output_stream_manager",
    srcs = ["output_stream_manager.cc"],
    hdrs = ["output_stream_manager.h"],
    visibility = [":mediapipe_internal"],
    deps = [
        ":input_stream_handler",
        ":output_stream_shard",
        ":packet",
        ":packet_type",
        ":port",
        ":timestamp",
        "//mediapipe/framework/port:source_location",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "output_stream_poller",
    hdrs = ["output_stream_poller.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_output_stream",
    ],
)

cc_library(
    name = "output_stream_shard",
    srcs = ["output_stream_shard.cc"],
    hdrs = ["output_stream_shard.h"],
    visibility = [":mediapipe_internal"],
    deps = [
        ":output_stream",
        ":packet",
        ":packet_type",
        ":port",
        ":timestamp",
        "//mediapipe/framework/port:source_location",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
    ],
)

# Defines Packet, a data carrier used throughout the framework.
cc_library(
    name = "packet",
    srcs = ["packet.cc"],
    hdrs = ["packet.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":port",
        ":timestamp",
        ":type_map",
        "//mediapipe/framework/deps:no_destructor",
        "//mediapipe/framework/deps:registration",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "//mediapipe/framework/tool:type_util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "packet_generator",
    hdrs = ["packet_generator.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":packet",
        ":packet_set",
        ":packet_type",
        ":port",
        "//mediapipe/framework:packet_generator_cc_proto",
        "//mediapipe/framework/deps:registration",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "packet_generator_graph",
    srcs = ["packet_generator_graph.cc"],
    hdrs = ["packet_generator_graph.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":delegating_executor",
        ":executor",
        ":packet",
        ":packet_generator",
        ":packet_type",
        ":port",
        ":thread_pool_executor",
        ":validated_graph_config",
        "//mediapipe/framework:packet_factory_cc_proto",
        "//mediapipe/framework:packet_generator_cc_proto",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:status_util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "packet_set",
    hdrs = ["packet_set.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":collection",
        ":packet",
    ],
)

cc_library(
    name = "packet_type",
    srcs = ["packet_type.cc"],
    hdrs = ["packet_type.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":collection",
        ":packet",
        ":packet_set",
        ":type_map",
        "//mediapipe/framework/deps:no_destructor",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:map_util",
        "//mediapipe/framework/port:source_location",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:status_util",
        "//mediapipe/framework/tool:type_util",
        "//mediapipe/framework/tool:validate_name",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

# When --copt=-fno-rtti is set, MEDIAPIPE_HAS_RTTI is cleared in port.h.
# To explicitly clear MEDIAPIPE_HAS_RTTI, compile with:
#   bazel build --define=disable_rtti_and_exceptions=true
config_setting(
    name = "disable_rtti_and_exceptions",
    define_values = {"disable_rtti_and_exceptions": "true"},
    visibility = ["//visibility:public"],
)

cc_library(
    name = "port",
    hdrs = ["port.h"],
    defines = select({
        "//conditions:default": [],
    }) + select({
        "//conditions:default": [],
        "//mediapipe/gpu:disable_gpu": ["MEDIAPIPE_DISABLE_GPU=1"],
    }) + select({
        "//conditions:default": [],
        "//mediapipe/framework/port:disable_opencv": ["MEDIAPIPE_DISABLE_OPENCV=1"],
    }) + select({
        "//conditions:default": [],
        "//mediapipe/framework:disable_rtti_and_exceptions": [
            "MEDIAPIPE_HAS_RTTI=0",
        ],
    }),
    visibility = [
        "//fitbit/research/sensing/mobisense:__subpackages__",
        "//mediapipe/calculators:__subpackages__",
        "//mediapipe/framework:__subpackages__",
        "//mediapipe/framework/port:__pkg__",
        "//mediapipe/gpu:__pkg__",
        "//mediapipe/util:__subpackages__",
    ],
)

cc_library(
    name = "scheduler_queue",
    srcs = ["scheduler_queue.cc"],
    hdrs = [
        "scheduler_queue.h",
        "scheduler_shared.h",
    ],
    copts = select({
        "//conditions:default": [],
        "//mediapipe:apple": [
            "-ObjC++",
        ],
    }),
    visibility = [":mediapipe_internal"],
    deps = [
        ":calculator_context",
        ":calculator_node",
        ":executor",
        "//mediapipe/framework/deps:clock",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "status_handler",
    hdrs = ["status_handler.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":packet_set",
        ":packet_type",
        ":port",
        "//mediapipe/framework:mediapipe_options_cc_proto",
        "//mediapipe/framework/deps:registration",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "subgraph",
    srcs = ["subgraph.cc"],
    hdrs = ["subgraph.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_service",
        ":graph_service_manager",
        ":port",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:mediapipe_options_cc_proto",
        "//mediapipe/framework/deps:registration",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "//mediapipe/framework/tool:calculator_graph_template_cc_proto",
        "//mediapipe/framework/tool:options_util",
        "//mediapipe/framework/tool:template_expander",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "test_calculators",
    testonly = 1,
    srcs = ["test_calculators.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":calculator_framework",
        "//mediapipe/framework:test_calculators_cc_proto",
        "//mediapipe/framework/deps:mathutil",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
    ],
    alwayslink = 1,
)

cc_library(
    name = "test_service",
    testonly = 1,
    srcs = ["test_service.cc"],
    hdrs = ["test_service.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":calculator_contract",
        ":calculator_framework",
        ":graph_service",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
)

cc_library(
    name = "thread_pool_executor",
    srcs = ["thread_pool_executor.cc"],
    hdrs = ["thread_pool_executor.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":executor",
        "//mediapipe/framework:thread_pool_executor_cc_proto",
        "//mediapipe/framework/deps:thread_options",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "//mediapipe/framework/port:threadpool",
        "//mediapipe/util:cpu_util",
    ],
)

cc_library(
    name = "timestamp",
    srcs = ["timestamp.cc"],
    hdrs = ["timestamp.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/deps:intops",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "throttler",
    hdrs = ["throttler.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "type_map",
    hdrs = ["type_map.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":demangle",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:status_util",
        "//mediapipe/framework/tool:type_util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = 1,
)

cc_library(
    name = "basic_types_registration",
    srcs = ["basic_types_registration.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":type_map",
        "//mediapipe/framework/port:integral_types",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "validated_graph_config",
    srcs = ["validated_graph_config.cc"],
    hdrs = ["validated_graph_config.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":calculator_base",
        ":calculator_contract",
        ":graph_service_manager",
        ":legacy_calculator_support",
        ":packet",
        ":packet_generator",
        ":packet_set",
        ":packet_type",
        ":port",
        ":status_handler",
        ":subgraph",
        ":timestamp",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:packet_generator_cc_proto",
        "//mediapipe/framework:status_handler_cc_proto",
        "//mediapipe/framework:stream_handler_cc_proto",
        "//mediapipe/framework:thread_pool_executor_cc_proto",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:map_util",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:source_location",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:topologicalsorter",
        "//mediapipe/framework/tool:name_util",
        "//mediapipe/framework/tool:status_util",
        "//mediapipe/framework/tool:subgraph_expansion",
        "//mediapipe/framework/tool:validate",
        "//mediapipe/framework/tool:validate_name",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "validated_graph_config_test",
    srcs = ["validated_graph_config_test.cc"],
    deps = [
        ":calculator_framework",
        ":graph_service",
        ":graph_service_manager",
        ":validated_graph_config",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/port:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "graph_validation",
    hdrs = ["graph_validation.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":calculator_framework",
        "//mediapipe/framework/port:status",
    ],
)

# cc tests
cc_test(
    name = "calculator_base_test",
    size = "medium",
    srcs = ["calculator_base_test.cc"],
    linkstatic = 1,
    deps = [
        ":calculator_base",
        ":calculator_context",
        ":calculator_context_manager",
        ":calculator_registry",
        ":calculator_state",
        ":output_stream",
        ":output_stream_manager",
        ":output_stream_shard",
        ":packet_set",
        ":packet_type",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:status_util",
        "//mediapipe/framework/tool:tag_map_helper",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_test(
    name = "calculator_contract_test",
    srcs = ["calculator_contract_test.cc"],
    linkstatic = 1,
    deps = [
        ":calculator_contract",
        ":calculator_contract_test_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:packet_generator_cc_proto",
        "//mediapipe/framework:status_handler_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
    ],
)

cc_test(
    name = "calculator_node_test",
    size = "small",
    srcs = ["calculator_node_test.cc"],
    linkstatic = 1,
    deps = [
        ":calculator_framework",
        ":calculator_node",
        "//mediapipe/calculators/core:pass_through_calculator",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:source",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "calculator_graph_event_loop_test",
    size = "small",
    srcs = ["calculator_graph_event_loop_test.cc"],
    deps = [
        ":calculator_framework",
        ":calculator_graph",
        "//mediapipe/calculators/core:pass_through_calculator",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:sink",
        "//mediapipe/framework/tool:status_util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "calculator_graph_stopping_test",
    size = "small",
    srcs = ["calculator_graph_stopping_test.cc"],
    deps = [
        ":calculator_framework",
        ":calculator_graph",
        "//mediapipe/calculators/core:pass_through_calculator",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:sink",
        "//mediapipe/framework/tool:status_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "calculator_parallel_execution_test",
    srcs = ["calculator_parallel_execution_test.cc"],
    deps = [
        ":calculator_framework",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:sink",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "calculator_runner_test",
    size = "medium",
    srcs = ["calculator_runner_test.cc"],
    deps = [
        ":calculator_base",
        ":calculator_registry",
        ":calculator_runner",
        ":input_stream",
        ":output_stream",
        ":packet_type",
        ":timestamp",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "calculator_context_test",
    size = "medium",
    srcs = ["calculator_context_test.cc"],
    linkstatic = 1,
    deps = [
        ":calculator_context",
        ":calculator_context_manager",
        ":calculator_state",
        ":output_stream",
        ":output_stream_manager",
        ":output_stream_shard",
        ":packet_set",
        ":packet_type",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/testdata:night_light_calculator_cc_proto",
        "//mediapipe/framework/testdata:sky_light_calculator_cc_proto",
        "//mediapipe/framework/tool:status_util",
        "//mediapipe/framework/tool:tag_map_helper",
    ],
)

cc_test(
    name = "calculator_graph_test",
    size = "small",
    srcs = [
        "calculator_graph_test.cc",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":calculator_framework",
        ":calculator_graph",
        ":collection_item_id",
        ":counter_factory",
        ":executor",
        ":input_stream_handler",
        ":lifetime_tracker",
        ":output_stream_poller",
        ":packet_set",
        ":packet_type",
        ":status_handler",
        ":subgraph",
        ":test_calculators",
        ":thread_pool_executor",
        ":timestamp",
        ":type_map",
        "//mediapipe/calculators/core:counting_source_calculator",
        "//mediapipe/calculators/core:mux_calculator",
        "//mediapipe/calculators/core:pass_through_calculator",
        "//mediapipe/framework:mediapipe_options_cc_proto",
        "//mediapipe/framework:thread_pool_executor_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/stream_handler:barrier_input_stream_handler",
        "//mediapipe/framework/stream_handler:early_close_input_stream_handler",
        "//mediapipe/framework/stream_handler:fixed_size_input_stream_handler",
        "//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        "//mediapipe/framework/stream_handler:mux_input_stream_handler",
        "//mediapipe/framework/stream_handler:sync_set_input_stream_handler",
        "//mediapipe/framework/stream_handler:timestamp_align_input_stream_handler",
        "//mediapipe/framework/tool:sink",
        "//mediapipe/framework/tool:status_util",
        "//mediapipe/gpu:graph_support",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "calculator_graph_bounds_test",
    size = "small",
    srcs = [
        "calculator_graph_bounds_test.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":calculator_context",
        ":calculator_framework",
        ":test_calculators",
        ":thread_pool_executor",
        ":timestamp",
        ":type_map",
        "//mediapipe/calculators/core:counting_source_calculator",
        "//mediapipe/calculators/core:mux_calculator",
        "//mediapipe/calculators/core:pass_through_calculator",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/stream_handler:barrier_input_stream_handler",
        "//mediapipe/framework/stream_handler:early_close_input_stream_handler",
        "//mediapipe/framework/stream_handler:fixed_size_input_stream_handler",
        "//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        "//mediapipe/framework/stream_handler:mux_input_stream_handler",
        "//mediapipe/framework/stream_handler:sync_set_input_stream_handler",
        "//mediapipe/framework/tool:sink",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "calculator_graph_side_packet_test",
    size = "small",
    srcs = [
        "calculator_graph_side_packet_test.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":calculator_framework",
        ":test_calculators",
        "//mediapipe/calculators/core:counting_source_calculator",
        "//mediapipe/calculators/core:mux_calculator",
        "//mediapipe/calculators/core:pass_through_calculator",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:sink",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "collection_test",
    size = "small",
    srcs = ["collection_test.cc"],
    linkstatic = 1,
    deps = [
        ":collection",
        ":packet_set",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:tag_map_helper",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "graph_service_test",
    size = "small",
    srcs = [
        "graph_service_test.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":calculator_contract",
        ":calculator_framework",
        ":graph_service",
        ":test_service",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:sink",
    ],
)

cc_test(
    name = "input_stream_manager_test",
    size = "small",
    srcs = ["input_stream_manager_test.cc"],
    linkstatic = 1,
    deps = [
        ":input_stream_manager",
        ":input_stream_shard",
        ":lifetime_tracker",
        ":packet",
        "//mediapipe/framework/port:gtest_main",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "output_stream_manager_test",
    size = "small",
    srcs = ["output_stream_manager_test.cc"],
    linkstatic = 1,
    deps = [
        ":input_stream_handler",
        ":input_stream_manager",
        ":output_stream_manager",
        ":output_stream_shard",
        ":packet",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/stream_handler:default_input_stream_handler",
        "//mediapipe/framework/tool:tag_map_helper",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "packet_delete_test",
    size = "small",
    srcs = ["packet_delete_test.cc"],
    copts = [
        "-Werror",
    ],
    linkstatic = 1,
    deps = [
        ":packet",
        "//mediapipe/framework/port:gtest_main",
    ],
)

cc_test(
    name = "executor_external_build_test",
    size = "small",
    srcs = ["executor_external_build_test.cc"],
    linkstatic = 1,
    deps = [
        ":executor",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:threadpool",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "packet_test",
    size = "medium",
    srcs = ["packet_test.cc"],
    linkstatic = 1,
    deps = [
        ":packet",
        ":packet_test_cc_proto",
        ":type_map",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:gtest_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "packet_registration_test",
    size = "small",
    srcs = ["packet_registration_test.cc"],
    deps = [
        ":calculator_framework",
        ":packet",
        ":packet_test_cc_proto",
        ":type_map",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:gtest_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "packet_generator_test",
    size = "small",
    srcs = ["packet_generator_test.cc"],
    deps = [
        ":packet_generator",
        ":packet_type",
        "//mediapipe/framework:packet_generator_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/tool:validate_type",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "timestamp_test",
    size = "small",
    srcs = ["timestamp_test.cc"],
    linkstatic = 1,
    deps = [
        ":timestamp",
        "//mediapipe/framework/port:gtest_main",
    ],
)

cc_test(
    name = "graph_validation_test",
    srcs = ["graph_validation_test.cc"],
    deps = [
        ":calculator_contract_test_cc_proto",
        ":calculator_framework",
        ":graph_validation",
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:default_side_packet_calculator",
        "//mediapipe/calculators/core:pass_through_calculator",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:packet_generator_cc_proto",
        "//mediapipe/framework:status_handler_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/tool:template_parser",
    ],
)

cc_test(
    name = "subgraph_test",
    srcs = ["subgraph_test.cc"],
    deps = [
        ":calculator_framework",
        ":graph_service_manager",
        ":subgraph",
        ":test_calculators",
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:pass_through_calculator",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/tool:sink",
        "//mediapipe/framework/tool/testdata:dub_quad_test_subgraph",
        "@com_google_absl//absl/strings:str_format",
    ],
)

# Expose the proto source files for building mediapipe AAR.
filegroup(
    name = "protos_src",
    srcs = glob(["*.proto"]),
    visibility = ["//mediapipe:__subpackages__"],
)
