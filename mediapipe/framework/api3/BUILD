load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//mediapipe:__subpackages__"] + [
])

licenses(["notice"])

cc_library(
    name = "calculator_context",
    hdrs = ["calculator_context.h"],
    deps = [
        ":contract",
        ":packet",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:graph_service",
        "//mediapipe/framework:resources",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/api3/internal:contract_fields",
        "//mediapipe/framework/api3/internal:contract_to_tuple",
        "//mediapipe/framework/api3/internal:port_base",
        "//mediapipe/framework/api3/internal:specializers",
        "@com_google_absl//absl/log:absl_log",
    ],
)

cc_test(
    name = "calculator_context_test",
    srcs = ["calculator_context_test.cc"],
    deps = [
        ":calculator_context",
        ":port_test_nodes",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "calculator_contract",
    hdrs = ["calculator_contract.h"],
    deps = [
        ":any",
        ":contract",
        "//mediapipe/framework:calculator_contract",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:graph_service",
        "//mediapipe/framework/api3/internal:contract_fields",
        "//mediapipe/framework/api3/internal:contract_to_tuple",
        "//mediapipe/framework/api3/internal:has_update_contract",
        "//mediapipe/framework/api3/internal:port_base",
        "//mediapipe/framework/api3/internal:specializers",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "calculator_contract_test",
    srcs = ["calculator_contract_test.cc"],
    deps = [
        ":calculator_contract",
        ":contract",
        ":node",
        ":port_test_nodes",
        "//mediapipe/framework:calculator_contract",
        "//mediapipe/framework:packet_type",
        "//mediapipe/framework/api3/internal:specializers",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "packet",
    hdrs = ["packet.h"],
    deps = [
        "//mediapipe/framework:packet",
        "//mediapipe/framework:timestamp",
        "@com_google_absl//absl/functional:any_invocable",
    ],
)

cc_library(
    name = "packet_matcher",
    testonly = True,
    hdrs = ["packet_matcher.h"],
    deps = [
        ":packet",
        "//mediapipe/framework:demangle",
        "//mediapipe/framework/port:gtest_main",
    ],
)

cc_test(
    name = "packet_test",
    srcs = ["packet_test.cc"],
    deps = [
        ":packet",
        ":packet_matcher",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:gtest_main",
    ],
)

cc_library(
    name = "contract",
    hdrs = ["contract.h"],
    deps = [
        "//mediapipe/framework/api3/internal:contract_fields",
        "//mediapipe/framework/api3/internal:port_base",
        "//mediapipe/framework/api3/internal:specializers",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
    ],
)

cc_library(
    name = "node",
    hdrs = ["node.h"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "node_test",
    srcs = [
        "node_test.cc",
        "node_test.h",
    ],
    deps = [
        ":contract",
        ":node",
        "//mediapipe/framework/api3/testing:bar_cc_proto",
        "//mediapipe/framework/api3/testing:foo_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "graph",
    hdrs = ["graph.h"],
    deps = [
        ":contract",
        ":side_packet",
        ":stream",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/api3/internal:contract_fields",
        "//mediapipe/framework/api3/internal:contract_to_tuple",
        "//mediapipe/framework/api3/internal:contract_validator",
        "//mediapipe/framework/api3/internal:graph_builder",
        "//mediapipe/framework/api3/internal:port_base",
        "//mediapipe/framework/api3/internal:specializers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "graph_test",
    srcs = [
        "graph_test.cc",
    ],
    deps = [
        ":contract",
        ":graph",
        ":node",
        ":side_packet",
        ":stream",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api3/testing:generator_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/stream_handler:fixed_size_input_stream_handler_cc_proto",
        "//mediapipe/framework/testdata:night_light_calculator_cc_proto",
        "//mediapipe/framework/testdata:sky_light_calculator_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "graph_interop_test",
    srcs = [
        "graph_interop_test.cc",
    ],
    deps = [
        ":contract",
        ":graph",
        ":port_test_nodes",
        ":side_packet",
        ":stream",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api3/testing:generator_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/stream_handler:fixed_size_input_stream_handler_cc_proto",
        "//mediapipe/framework/testdata:night_light_calculator_cc_proto",
        "//mediapipe/framework/testdata:sky_light_calculator_cc_proto",
    ],
)

cc_library(
    name = "calculator",
    hdrs = ["calculator.h"],
    deps = [
        ":calculator_context",
        ":calculator_contract",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api3/internal:contract_validator",
        "//mediapipe/framework/api3/internal:has_update_contract",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "calculator_test",
    srcs = [
        "calculator_test.cc",
        "calculator_test.h",
    ],
    deps = [
        ":calculator",
        ":calculator_context",
        ":calculator_contract",
        ":contract",
        ":node",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:graph_service",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/api3/testing:bar_cc_proto",
        "//mediapipe/framework/api3/testing:foo_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/tool:options_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "port_test_nodes",
    testonly = True,
    hdrs = ["port_test_nodes.h"],
    deps = [
        ":contract",
        ":node",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "stream",
    hdrs = ["stream.h"],
    deps = [
        ":any",
        "//mediapipe/framework/api3/internal:graph_builder",
    ],
)

cc_library(
    name = "side_packet",
    hdrs = ["side_packet.h"],
    deps = [
        ":any",
        "//mediapipe/framework/api3/internal:graph_builder",
    ],
)

cc_library(
    name = "any",
    hdrs = ["any.h"],
)

cc_test(
    name = "any_test",
    srcs = [
        "any_test.cc",
    ],
    deps = [
        ":any",
        ":calculator",
        ":calculator_context",
        ":calculator_contract",
        ":contract",
        ":graph",
        ":node",
        ":side_packet",
        ":stream",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/tool:options_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)
