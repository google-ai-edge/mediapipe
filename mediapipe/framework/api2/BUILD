load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")
# Placeholder: load py_test

package(
    default_visibility = ["//visibility:public"],
    features = ["-use_header_modules"],
)

licenses(["notice"])

cc_library(
    name = "const_str",
    hdrs = ["const_str.h"],
)

cc_library(
    name = "builder",
    hdrs = ["builder.h"],
    deps = [
        ":packet",
        ":port",
        "//mediapipe/framework:calculator_base",
        "//mediapipe/framework:calculator_contract",
        "//mediapipe/framework:mediapipe_options_cc_proto",
        "//mediapipe/framework:stream_handler_cc_proto",
        "//mediapipe/framework/port:any_proto",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "builder_test",
    srcs = ["builder_test.cc"],
    deps = [
        ":builder",
        ":node",
        ":packet",
        ":port",
        ":tag",
        ":test_contracts",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/stream_handler:fixed_size_input_stream_handler_cc_proto",
        "//mediapipe/framework/testdata:night_light_calculator_cc_proto",
        "//mediapipe/framework/testdata:sky_light_calculator_cc_proto",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "contract",
    hdrs = ["contract.h"],
    deps = [
        ":const_str",
        ":packet",
        ":port",
        ":tag",
        ":tuple",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_contract",
        "//mediapipe/framework:output_side_packet",
        "//mediapipe/framework/port:logging",
    ],
)

cc_test(
    name = "contract_test",
    srcs = ["contract_test.cc"],
    deps = [
        ":contract",
        ":port",
        ":tag",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
    ],
)

cc_library(
    name = "node",
    srcs = ["node.cc"],
    hdrs = ["node.h"],
    deps = [
        ":contract",
        "//mediapipe/framework:calculator_base",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_contract",
        "//mediapipe/framework:subgraph",
        "//mediapipe/framework/deps:registration",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "test_contracts",
    testonly = True,
    hdrs = ["test_contracts.h"],
    deps = [
        ":node",
    ],
)

cc_test(
    name = "node_test",
    srcs = ["node_test.cc"],
    deps = [
        ":builder",
        ":contract",
        ":node",
        ":node_test_cc_proto",
        ":packet",
        ":port",
        ":test_contracts",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

mediapipe_proto_library(
    name = "node_test_proto",
    srcs = ["node_test.proto"],
)

cc_library(
    name = "packet",
    srcs = ["packet.cc"],
    hdrs = ["packet.h"],
    deps = [
        ":tuple",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "packet_test",
    size = "small",
    srcs = [
        "packet_test.cc",
    ],
    deps = [
        ":packet",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "port",
    hdrs = ["port.h"],
    deps = [
        ":const_str",
        ":packet",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_contract",
        "//mediapipe/framework:output_side_packet",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/tool:type_util",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "port_test",
    size = "small",
    srcs = [
        "port_test.cc",
    ],
    deps = [
        ":builder",
        ":node",
        ":packet",
        ":port",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "subgraph_test",
    srcs = ["subgraph_test.cc"],
    deps = [
        ":builder",
        ":node",
        ":packet",
        ":port",
        ":test_contracts",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/tool:subgraph_expansion",
    ],
)

cc_library(
    name = "tag",
    hdrs = ["tag.h"],
    deps = [":const_str"],
)

cc_test(
    name = "tag_test",
    size = "small",
    srcs = [
        "tag_test.cc",
    ],
    deps = [
        ":tag",
        "//mediapipe/framework/port:gtest_main",
    ],
)

cc_library(
    name = "tuple",
    hdrs = ["tuple.h"],
    deps = ["@com_google_absl//absl/meta:type_traits"],
)

cc_test(
    name = "tuple_test",
    size = "small",
    srcs = [
        "tuple_test.cc",
    ],
    deps = [
        ":tuple",
        "//mediapipe/framework/port:gtest_main",
        "@com_google_absl//absl/strings",
    ],
)
