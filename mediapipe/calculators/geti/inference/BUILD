load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library", "mediapipe_proto_library")
load("@rules_cc//cc:defs.bzl", "cc_proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "grpc_predict_v2_proto",
    srcs = ["grpc_predict_v2.proto"],
)

cc_proto_library(
    name = "grpc_predict_v2_cc_proto",
    deps = [":grpc_predict_v2_proto"],
)

proto_library(
    name = "openvino_inference_adapter_calculator_proto",
    srcs = ["openvino_inference_adapter_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",

    ],
)

mediapipe_cc_proto_library(
    name = "openvino_inference_adapter_calculator_cc_proto",
    srcs = ["openvino_inference_adapter_calculator.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":openvino_inference_adapter_calculator_proto"],
)

cc_library(
    name = "inference_calculators",
    visibility = [
        "//visibility:public",
    ],
    srcs = [
        "anomaly_calculator.cc",
        "classification_calculator.cc",
        "detection_calculator.cc",
        "instance_segmentation_calculator.cc",
        "openvino_inference_adapter_calculator.cc",
        "rotated_detection_calculator.cc",
        "segmentation_calculator.cc",
        "model_infer_request_image_calculator.cc",
    ],
    hdrs = [
        "anomaly_calculator.h",
        "classification_calculator.h",
        "detection_calculator.h",
        "instance_segmentation_calculator.h",
        "kserve.h",
        "openvino_inference_adapter_calculator.h",
        "rotated_detection_calculator.h",
        "segmentation_calculator.h",
        "model_infer_request_image_calculator.h",
    ],
    deps = [
        ":openvino_inference_adapter_calculator_cc_proto",
        ":grpc_predict_v2_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/calculators/geti/utils:data_structures",
        "//mediapipe/calculators/geti/utils:emptylabel_cc_proto",
        ":inference_utils",
        "@model_api//:model_api",
        ":geti_calculator_base",
    ],
    defines = [
        "USE_MODELADAPTER",
    ],
    alwayslink = 1,
)

cc_library(
    name = "inference_utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/calculators/geti/utils:data_structures",
        "@model_api//:model_api",
    ]
)

cc_library(
    name = "geti_calculator_base",
    hdrs = [
        "geti_calculator_base.h",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
    ],
    visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "test_deps",
    hdrs = [ "test_utils.h" ],
    deps = [
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/util:image_test_utils",
        "@nlohmann_json//:json",
    ],
    visibility = [
        "//visibility:public",
    ],
    testonly = 1
)

cc_test(
    name = "inference_tests",
    srcs = [
        "anomaly_calculator_test.cc",
        "classification_calculator_test.cc",
        "detection_calculator_test.cc",
        "rotated_detection_calculator_test.cc",
        "segmentation_calculator_test.cc",
        "instance_segmentation_calculator_test.cc",
    ],
    deps = [
        ":inference_calculators",
        ":test_deps"
    ],
    defines = [
        "USE_MODELADAPTER",
    ],
)

cc_test(
    name = "model_infer_request_image_calculator_test",
    srcs = [
        "model_infer_request_image_calculator_test.cc",
    ],
    deps = [
        ":inference_calculators",
        ":test_deps"
    ],
    defines = [
        "USE_MODELADAPTER",
    ],
)
