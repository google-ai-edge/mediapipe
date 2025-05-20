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
        ":specializers",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_contract",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)
