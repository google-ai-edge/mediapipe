load("@bazel_skylib//lib:selects.bzl", "selects")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//mediapipe/framework:mediapipe_cc_test.bzl", "mediapipe_cc_test")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

# This can be used to force WebGPU on in desktop builds.
config_setting(
    name = "use_webgpu_define",
    define_values = {"MEDIAPIPE_USE_WEBGPU": "1"},
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "use_webgpu",
    match_any = [
        ":use_webgpu_define",
    ],
)

selects.config_setting_group(
    name = "use_webgpu_emscripten",
    match_all = [
        ":use_webgpu_define",
        "//mediapipe:emscripten",
    ],
)

exports_files([
    "tensor_webgpu_texture_view.cc",
])

cc_library(
    name = "webgpu_device_registration",
    srcs = select({
        "//mediapipe:emscripten": [],
        "//conditions:default": ["webgpu_device_registration.cc"],
    }),
    hdrs = select({
        "//mediapipe:emscripten": [],
        "//conditions:default": ["webgpu_device_registration.h"],
    }),
    deps = select({
        "//mediapipe:emscripten": [],
        "//conditions:default": [
            "//mediapipe/framework/deps:no_destructor",
            "//third_party/dawn:dawncpp_headers",
        ],
    }),
)

cc_library(
    name = "webgpu_check",
    srcs = ["webgpu_check.cc"],
    hdrs = ["webgpu_check.h"],
    deps = select({
        # `:webgpu_device_registration` should only be included if we are on a non-web platform, and
        # `MEDIAPIPE_USE_WEBGPU` is set to true. That is why we added both `:use_webgpu` and
        # `:use_webgpu_define`. Because in `select`, if multiple conditions match and one is a
        # specialization of the others, the specialization takes precedence. So, `:use_webgpu` will
        # filter out `emscripten`.
        ":use_webgpu_define": [":webgpu_device_registration"],
        ":use_webgpu": [],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "webgpu_service",
    srcs = ["webgpu_service.cc"],
    hdrs = ["webgpu_service.h"],
    deps = [
        ":webgpu_check",
        "//mediapipe/framework:graph_service",
        "//mediapipe/framework/deps:no_destructor",
        "//mediapipe/gpu:attachments",
        "//third_party/dawn:webgpu_headers",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ] + select({
        "//mediapipe:emscripten": [
        ],
        "//conditions:default": [
            ":webgpu_device_registration",
        ],
    }),
)

cc_library(
    name = "webgpu_texture_view",
    hdrs = ["webgpu_texture_view.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/gpu:gpu_buffer_storage",
    ] + select({
        "//mediapipe:emscripten": [],
        "//conditions:default": [
            "//third_party/dawn:webgpu_headers",
        ],
    }),
)

cc_library(
    name = "webgpu_texture_buffer",
    srcs = ["webgpu_texture_buffer.cc"],
    hdrs = ["webgpu_texture_buffer.h"],
    tags = ["keep_dep"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":webgpu_service",
        ":webgpu_texture_view",
        "//mediapipe/framework:legacy_calculator_support",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/gpu:gpu_buffer_format",
        "//mediapipe/gpu:gpu_buffer_storage",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = 1,
)

cc_library(
    name = "webgpu_utils",
    srcs = ["webgpu_utils.cc"],
    hdrs = ["webgpu_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/port:status",
        "//mediapipe/web:jspi_check",
        "//third_party/dawn:webgpu_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
    ] + select({
        "//mediapipe:emscripten": [
            "//third_party/emdawnwebgpu:webgpu",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "image_to_tensor_converter_webgpu_texture",
    srcs = ["image_to_tensor_converter_webgpu_texture.cc"],
    hdrs = ["image_to_tensor_converter_webgpu_texture.h"],
    deps = [
        ":webgpu_service",
        ":webgpu_texture_view",
        ":webgpu_utils",
        "//mediapipe/calculators/tensor:image_to_tensor_converter",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:tensor",
    ],
)

cc_library(
    name = "webgpu_shader_calculator",
    srcs = ["webgpu_shader_calculator.cc"],
    deps = [
        ":webgpu_shader_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/deps:re2",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/gpu:gpu_buffer_format",
        "//mediapipe/gpu/webgpu:webgpu_service",
        "//mediapipe/gpu/webgpu:webgpu_texture_buffer",
        "//mediapipe/gpu/webgpu:webgpu_texture_buffer_3d",
        "//mediapipe/gpu/webgpu:webgpu_texture_view",
        "//mediapipe/gpu/webgpu:webgpu_utils",
        "//mediapipe/util:resource_util",
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "webgpu_shader_calculator_proto",
    srcs = ["webgpu_shader_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)
