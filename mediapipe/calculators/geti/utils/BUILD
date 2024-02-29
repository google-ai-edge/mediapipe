licenses(["notice"])
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library", "mediapipe_proto_library")


package(default_visibility = ["//visibility:public"])

cc_library(
    name = "utils",
    deps = [
        ":data_structures",
        ":loop_calculators",
        ":crop_calculator",
        ":overlay_calculator",
        ":detection_extraction_calculator",
        ":combiner_calculators",
        ":emptylabel_calculators"
    ],
    alwayslink = 1,
)

cc_library(
    name = "data_structures",
    hdrs = [
        "data_structures.h",
    ],
    deps = [
        "@model_api//:model_api",
    ]
)

cc_library(
    name = "detection_extraction_calculator",
    srcs = [
        "detection_extraction_calculator.cc",
    ],
    hdrs = [
        "detection_extraction_calculator.h",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        ":data_structures",
        "//mediapipe/calculators/geti/inference:geti_calculator_base",
        "//mediapipe/calculators/geti/inference:inference_utils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "loop_calculators",
    srcs = [
        "loop_calculators.cc",
    ],
    hdrs = [
        "loop_calculators.h",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/calculators/core:begin_loop_calculator",
        "//mediapipe/calculators/core:end_loop_calculator",
        ":data_structures",
        "//mediapipe/calculators/geti/inference:geti_calculator_base",
        "@model_api//:model_api",
    ],
    alwayslink = 1,
)

cc_library(
    name = "combiner_calculators",
    srcs = [
        "detection_classification_combiner_calculator.cc",
        "detection_classification_result_calculator.cc",
        "detection_segmentation_combiner_calculator.cc",
        "detection_segmentation_result_calculator.cc"
    ],
    hdrs = [
        "detection_classification_combiner_calculator.h",
        "detection_classification_result_calculator.h",
        "detection_segmentation_combiner_calculator.h",
        "detection_segmentation_result_calculator.h"
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        ":data_structures",
        "//mediapipe/calculators/geti/inference:geti_calculator_base",
        "//mediapipe/calculators/geti/inference:inference_utils",
    ],
    alwayslink = 1
)

cc_library(
    name = "crop_calculator",
    srcs = [
        "crop_calculator.cc",
    ],
    hdrs = [
        "crop_calculator.h",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_core",
        ":data_structures",
        "//mediapipe/calculators/geti/inference:geti_calculator_base",
    ],
    alwayslink = 1,
)

cc_test(
    name = "crop_calculator_test",
    srcs = [
        "crop_calculator_test.cc",
    ],
    deps = [
        ":crop_calculator",
        "//mediapipe/calculators/geti/inference:test_deps"
    ]
)

cc_library(
    name = "overlay_calculator",
    srcs = [
        "overlay_calculator.cc",
    ],
    hdrs = [
        "overlay_calculator.h",
    ],
    deps = [
        ":data_structures",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_core",
        "@model_api//:model_api",
        "//mediapipe/calculators/geti/inference:geti_calculator_base",
    ],
    alwayslink = 1,
)

proto_library(
    name = "emptylabel_proto",
    srcs = ["emptylabel.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "emptylabel_cc_proto",
    srcs = ["emptylabel.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":emptylabel_proto"],
)

cc_library(
    name = "emptylabel_calculators",
    srcs = [
        "emptylabel_calculator.cc",
    ],
    hdrs = [
        "emptylabel_calculator.h",
    ],
    deps = [
        ":data_structures",
        ":emptylabel_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_core",
        "@model_api//:model_api",
        "//mediapipe/calculators/geti/inference:geti_calculator_base",
    ],
    alwayslink = 1,
)

cc_test(
    name = "emptylabel_calculator_test",
   srcs = [
        "emptylabel_calculator_test.cc",
    ],
    deps = [
        ":emptylabel_calculators",
        "//mediapipe/calculators/geti/inference:test_deps"
    ],
    defines = [
       "USE_MODELADAPTER",
    ],
)

