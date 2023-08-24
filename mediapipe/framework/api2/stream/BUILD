package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "loopback",
    hdrs = ["loopback.h"],
    deps = [
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
    ],
)

cc_test(
    name = "loopback_test",
    srcs = ["loopback_test.cc"],
    deps = [
        ":loopback",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status_matchers",
    ],
)

cc_library(
    name = "image_size",
    hdrs = ["image_size.h"],
    deps = [
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/gpu:gpu_buffer",
    ],
)

cc_test(
    name = "image_size_test",
    srcs = ["image_size_test.cc"],
    deps = [
        ":image_size",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status_matchers",
        "//mediapipe/gpu:gpu_buffer",
    ],
)

cc_library(
    name = "rect_transformation",
    srcs = ["rect_transformation.cc"],
    hdrs = ["rect_transformation.h"],
    deps = [
        "//mediapipe/calculators/util:rect_transformation_calculator",
        "//mediapipe/calculators/util:rect_transformation_calculator_cc_proto",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/formats:rect_cc_proto",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_test(
    name = "rect_transformation_test",
    srcs = ["rect_transformation_test.cc"],
    deps = [
        ":rect_transformation",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
    ],
)
