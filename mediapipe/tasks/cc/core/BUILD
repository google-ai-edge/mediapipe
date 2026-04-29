# Copyright 2022 The MediaPipe Authors.

load("@org_tensorflow//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite", "cc_test_with_tflite")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# =========================
# 🔥 COMMON MACROS
# =========================

def tflite_lib(name, srcs = [], hdrs = [], deps = [], tflite_deps = [], alwayslink = False):
    cc_library_with_tflite(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        deps = deps,
        tflite_deps = tflite_deps,
        alwayslink = 1 if alwayslink else 0,
    )

# =========================
# 🔥 SHARED DEPS
# =========================

ABSL_BASE = [
    "@com_google_absl//absl/status",
    "@com_google_absl//absl/status:statusor",
    "@com_google_absl//absl/strings",
]

TFLITE_CORE = [
    "@org_tensorflow//tensorflow/lite/core/api:op_resolver",
]

# =========================
# 🔥 CORE LIBS
# =========================

cc_library(
    name = "host_environment",
    hdrs = ["host_environment.h"],
)

cc_library(
    name = "base_options",
    srcs = ["base_options.cc"],
    hdrs = ["base_options.h"],
    deps = [
        ":host_environment",
        ":mediapipe_builtin_op_resolver",
        "//mediapipe/calculators/tensor:inference_calculator_cc_proto",
        "//mediapipe/tasks/cc/core/proto:acceleration_cc_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_cc_proto",
        "//mediapipe/tasks/cc/core/proto:external_file_cc_proto",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
    ] + TFLITE_CORE,
)

cc_library(
    name = "external_file_handler",
    srcs = ["external_file_handler.cc"],
    hdrs = ["external_file_handler.h"],
    deps = [
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/core/proto:external_file_cc_proto",
        "//mediapipe/util:resource_util",
        "@com_google_absl//absl/memory",
    ] + ABSL_BASE + [
        "@com_google_absl//absl/strings:str_format",
    ],
)

# =========================
# 🔥 TFLITE LIBS
# =========================

tflite_lib(
    name = "mediapipe_builtin_op_resolver",
    srcs = ["mediapipe_builtin_op_resolver.cc"],
    hdrs = ["mediapipe_builtin_op_resolver.h"],
    tflite_deps = [
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
    deps = [
        "//mediapipe/tasks/cc/text/custom_ops/ragged:ragged_tensor_to_tensor_tflite",
        "//mediapipe/tasks/cc/text/custom_ops/sentencepiece:sentencepiece_tokenizer_tflite",
        "//mediapipe/tasks/cc/text/language_detector/custom_ops:kmeans_embedding_lookup",
        "//mediapipe/tasks/cc/text/language_detector/custom_ops:ngram_hash",
        "//mediapipe/tasks/cc/vision/custom_ops:fused_batch_norm",
        "//mediapipe/util/tflite/operations:landmarks_to_transform_matrix",
        "//mediapipe/util/tflite/operations:max_pool_argmax",
        "//mediapipe/util/tflite/operations:max_unpooling",
        "//mediapipe/util/tflite/operations:transform_landmarks",
        "//mediapipe/util/tflite/operations:transform_tensor_bilinear",
        "//mediapipe/util/tflite/operations:transpose_conv_bias",
    ],
)

tflite_lib(
    name = "model_resources",
    srcs = ["model_resources.cc"],
    hdrs = ["model_resources.h"],
    tflite_deps = [
        "@org_tensorflow//tensorflow/lite:framework_stable",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/tools:verifier",
    ],
    deps = [
        ":external_file_handler",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/core/proto:external_file_cc_proto",
        "//mediapipe/tasks/cc/metadata:metadata_extractor",
        "//mediapipe/util:resource_util",
        "//mediapipe/util:resource_util_custom",
        "//mediapipe/util/tflite:error_reporter",
        "@com_google_absl//absl/memory",
    ] + ABSL_BASE + TFLITE_CORE + [
        "@org_tensorflow//tensorflow/lite/core/api:error_reporter",
    ],
)

tflite_lib(
    name = "model_resources_cache",
    srcs = ["model_resources_cache.cc"],
    hdrs = ["model_resources_cache.h"],
    tflite_deps = [
        ":model_resources",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
    deps = [
        ":model_asset_bundle_resources",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/tasks/cc:common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
    ] + ABSL_BASE + TFLITE_CORE,
)

tflite_lib(
    name = "model_resources_calculator",
    srcs = ["model_resources_calculator.cc"],
    tflite_deps = [
        ":model_resources",
        ":model_resources_cache",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:port",
        "//mediapipe/tasks/cc/core/proto:external_file_cc_proto",
        "//mediapipe/tasks/cc/core/proto:model_resources_calculator_cc_proto",
        "//mediapipe/tasks/cc/metadata:metadata_extractor",
    ] + ABSL_BASE + TFLITE_CORE,
    alwayslink = True,
)

tflite_lib(
    name = "task_runner",
    srcs = ["task_runner.cc"],
    hdrs = ["task_runner.h"],
    tflite_deps = [
        ":model_resources",
        ":model_resources_cache",
        ":model_resources_calculator",
    ],
    deps = [
        ":host_environment",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:executor",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:name_util",
        "//mediapipe/gpu:gpu_shared_data_internal",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/core/logging:tasks_dummy_logger",
        "//mediapipe/tasks/cc/core/logging:tasks_logger",
        "//mediapipe/tasks/cc/core/logging/factory:logging_factory",
        "//mediapipe/tasks/cc/core/logging/factory:logging_factory_hdr",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
    ] + ABSL_BASE + TFLITE_CORE,
)

tflite_lib(
    name = "base_task_api",
    hdrs = ["base_task_api.h"],
    tflite_deps = [":task_runner"],
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator",
    ],
)

tflite_lib(
    name = "task_api_factory",
    hdrs = ["task_api_factory.h"],
    tflite_deps = [
        ":base_task_api",
        ":model_resources",
        ":task_runner",
    ],
    deps = [
        ":utils",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:executor",
        "//mediapipe/framework/port:requires",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/core/proto:base_options_cc_proto",
        "//mediapipe/tasks/cc/core/proto:external_file_cc_proto",
        "//mediapipe/tasks/cc/core/proto:inference_subgraph_cc_proto",
    ] + ABSL_BASE + TFLITE_CORE,
)

# =========================
# 🔥 NORMAL LIBS
# =========================

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    visibility = [
        "//mediapipe/calculators:__subpackages__",
        "//mediapipe/tasks:internal",
    ],
    deps = [
        ":external_file_handler",
        "//mediapipe/calculators/core:flow_limiter_calculator_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/tasks/cc/core/proto:external_file_cc_proto",
        "//mediapipe/tasks/metadata:metadata_schema_cc",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "model_asset_bundle_resources",
    srcs = ["model_asset_bundle_resources.cc"],
    hdrs = ["model_asset_bundle_resources.h"],
    deps = [
        ":external_file_handler",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/core/proto:external_file_cc_proto",
        "//mediapipe/tasks/cc/metadata/utils:zip_utils",
        "//mediapipe/util:resource_util",
        "@com_google_absl//absl/container:flat_hash_map",
    ] + ABSL_BASE + [
        "@com_google_absl//absl/strings:str_format",
    ],
)
