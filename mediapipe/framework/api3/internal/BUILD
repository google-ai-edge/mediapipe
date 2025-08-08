load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//mediapipe/framework/api3:__subpackages__"])

licenses(["notice"])

cc_library(
    name = "dependent_false",
    hdrs = ["dependent_false.h"],
)

cc_library(
    name = "contract_to_tuple",
    hdrs = ["contract_to_tuple.h"],
    deps = [":dependent_false"],
)

cc_test(
    name = "contract_to_tuple_test",
    srcs = ["contract_to_tuple_test.cc"],
    deps = [
        ":contract_to_tuple",
        ":specializers",
        "//mediapipe/framework/api3:calculator_contract",
        "//mediapipe/framework/api3:contract",
        "//mediapipe/framework/port:gtest_main",
    ],
)

cc_library(
    name = "contract_validator",
    hdrs = ["contract_validator.h"],
    deps = [
        ":contract_fields",
        ":contract_to_tuple",
        ":dependent_false",
        ":specializers",
        "//mediapipe/framework/api3:calculator_contract",
        "//mediapipe/framework/deps:registration",
        "//mediapipe/framework/tool:status_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "contract_validator_test",
    srcs = ["contract_validator_test.cc"],
    deps = [
        ":contract_validator",
        "//mediapipe/framework/api3:contract",
        "//mediapipe/framework/api3/testing:foo_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "has_update_contract",
    hdrs = ["has_update_contract.h"],
)

cc_test(
    name = "has_update_contract_test",
    srcs = ["has_update_contract_test.cc"],
    deps = [
        ":has_update_contract",
        "//mediapipe/framework/port:gtest_main",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "for_each_on_tuple_pair",
    hdrs = ["for_each_on_tuple_pair.h"],
)

cc_test(
    name = "for_each_on_tuple_pair_test",
    srcs = ["for_each_on_tuple_pair_test.cc"],
    deps = [
        ":for_each_on_tuple_pair",
        "//mediapipe/framework/port:gtest_main",
    ],
)

cc_library(
    name = "contract_fields",
    hdrs = ["contract_fields.h"],
)

cc_library(
    name = "specializers",
    hdrs = ["specializers.h"],
)

cc_library(
    name = "port_base",
    hdrs = ["port_base.h"],
    deps = [
        ":contract_fields",
        ":dependent_false",
        ":graph_builder",
        ":specializers",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_contract",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api3:any",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "graph_builder",
    hdrs = ["graph_builder.h"],
    visibility = [
        "//mediapipe/framework/api2:__subpackages__",
        "//mediapipe/framework/api3:__subpackages__",
    ],
    deps = [
        "//mediapipe/framework:calculator_base",
        "//mediapipe/framework:calculator_contract",
        "//mediapipe/framework:mediapipe_options_cc_proto",
        "//mediapipe/framework:stream_handler_cc_proto",
        "//mediapipe/framework/port:any_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:type_util",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "graph_builder_test",
    srcs = ["graph_builder_test.cc"],
    deps = [
        ":graph_builder",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/stream_handler:fixed_size_input_stream_handler_cc_proto",
        "//mediapipe/framework/testdata:night_light_calculator_cc_proto",
        "//mediapipe/framework/testdata:sky_light_calculator_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)
