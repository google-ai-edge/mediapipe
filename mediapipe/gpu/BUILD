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

load("@bazel_skylib//lib:selects.bzl", "selects")  # buildifier: disable=out-of-order-load
load("//mediapipe/gpu:metal.bzl", "metal_library")
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_unit_test")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")  # buildifier: disable=out-of-order-load
load("//mediapipe/framework:mediapipe_cc_test.bzl", "mediapipe_cc_test")  # buildifier: disable=out-of-order-load
load("//mediapipe/framework:more_selects.bzl", "more_selects")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

# Disabling GPU support is sometimes useful on desktop Linux because SwiftShader can
# interfere with desktop GL. b/73494271
config_setting(
    name = "disable_gpu",
    define_values = {
        "MEDIAPIPE_DISABLE_GPU": "1",
    },
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gpu_service",
    srcs = ["gpu_service.cc"],
    hdrs = ["gpu_service.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:graph_service",
        "@com_google_absl//absl/base:core_headers",
    ] + select({
        "//conditions:default": [
            ":gpu_shared_data_internal",
        ],
        "//mediapipe/gpu:disable_gpu": [],
    }),
)

cc_library(
    name = "graph_support",
    hdrs = ["graph_support.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "attachments",
    hdrs = ["attachments.h"],
    visibility = ["//visibility:public"],
)

GL_BASE_LINK_OPTS = select({
    "//conditions:default": [],
    "//mediapipe:android": [
        "-lGLESv2",
        "-lEGL",
        # Note: on Android, libGLESv3.so is normally a symlink to
        # libGLESv2.so, so we don't need to link to it. In fact, we
        # do not _want_ to link to it, or we would be unable to load
        # on API level < 18, where the symlink is missing entirely.
        # Note: if we ever find a strange version of Android where the
        # GLESv3 library is not a symlink, we will have to load it at
        # runtime. Weak GLESv3 symbols will still be resolved if we
        # load it early enough.
    ],
    "//mediapipe:ios": [
        "-framework OpenGLES",
        "-framework CoreVideo",
    ],
    "//mediapipe:macos": [
        "-framework OpenGL",
        "-framework CoreVideo",
    ],
})

# This is @unused internally.
GL_BASE_LINK_OPTS_OSS = GL_BASE_LINK_OPTS + select({
    "//conditions:default": [
        # Use GLES/EGL on linux.
        # Requires support from graphics card driver (nvidia,mesa,etc..)
        # and libraries to be installed.
        # Ex: libegl1-mesa-dev libgles2-mesa-dev, or libegl1-nvidia libgles2-nvidia, etc...
        "-lGLESv2",
        "-lEGL",
    ],
    "//mediapipe:android": [],
    ":platform_apple_with_gpu": [],
    ":disable_gpu": [],
})

cc_library(
    name = "gl_base",
    defines = select({
        "//mediapipe:apple": [
            "GLES_SILENCE_DEPRECATION=1",
        ],
        "//conditions:default": [],
    }),
    features = ["-layering_check"],
    linkopts = GL_BASE_LINK_OPTS_OSS,
    textual_hdrs = ["gl_base.h"],
    visibility = ["//visibility:public"],
    deps = [":gl_base_hdr"] + select({
        "//mediapipe:android": [],
        "//mediapipe:apple": [],
        "//conditions:default": [
        ],
    }),
)

cc_library(
    name = "gl_base_hdr",
    hdrs = ["gl_base.h"],
    features = ["-layering_check"],
    # Note: need the frameworks on Apple platforms to get the headers.
    linkopts = select({
        "//conditions:default": [],
        "//mediapipe:ios": [
            "-framework OpenGLES",
            "-framework CoreVideo",
        ],
        "//mediapipe:macos": [
            "-framework OpenGL",
            "-framework CoreVideo",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = select({
        "//mediapipe:android": [],
        "//mediapipe:apple": [],
        "//conditions:default": [
        ],
    }),
)

cc_library(
    name = "gl_thread_collector",
    hdrs = ["gl_thread_collector.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gl_base",
    ],
)

cc_library(
    name = "gl_context",
    srcs = [
        "gl_context.cc",
        "gl_context_internal.h",
    ] + select({
        "//conditions:default": [
            "gl_context_egl.cc",
        ],
        "//mediapipe:ios": [
            "gl_context_eagl.cc",
        ],
        "//mediapipe:macos": [
            "gl_context_nsgl.cc",
        ],
    }),
    hdrs = ["gl_context.h"],
    copts = select({
        "//conditions:default": [],
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
    }),
    linkopts = select({
        "//conditions:default": [],
        "//mediapipe:ios": [
            "-framework OpenGLES",
        ],
        "//mediapipe:macos": [
            "-framework OpenGL",
            "-framework AppKit",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":attachments",
        ":gl_base",
        ":gl_thread_collector",
        ":gpu_buffer_format",
        "//mediapipe/framework:executor",
        "//mediapipe/framework:mediapipe_profiling",
        "//mediapipe/framework:port",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "//mediapipe/framework/port:threadpool",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
    ] + select({
        "//conditions:default": [],
        "//mediapipe:apple": [
            "//mediapipe/objc:CFHolder",
        ],
    }) + select({
        "//conditions:default": [
        ],
        "//mediapipe:ios": [
        ],
        "//mediapipe:macos": [
        ],
    }),
)

cc_library(
    name = "gl_texture_buffer",
    srcs = ["gl_texture_buffer.cc"],
    hdrs = ["gl_texture_buffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gl_base",
        ":gl_context",
        ":gl_texture_view",
        ":gpu_buffer_format",
        ":gpu_buffer_storage",
        ":gpu_buffer_storage_image_frame",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        # TODO: remove this dependency. Some other teams' tests
        # depend on having an indirect image_frame dependency, need to be
        # fixed first.
        "//mediapipe/framework/formats:image_frame",
    ] + select({
        "//conditions:default": [],
        ":platform_apple_with_gpu": [
            ":gl_texture_util",
            ":gpu_buffer_storage_cv_pixel_buffer",
        ],
    }),
)

cc_library(
    name = "gl_texture_view",
    srcs = ["gl_texture_view.cc"],
    hdrs = ["gl_texture_view.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gl_base",
        ":gl_context",
        ":gpu_buffer_storage",
    ],
)

# Workaround for "Multiple matches are not allowed unless one is unambiguously more specialized".
more_selects.config_setting_negation(
    name = "not_disable_gpu",
    negate = ":disable_gpu",
)

selects.config_setting_group(
    name = "platform_ios_with_gpu",
    match_all = [
        ":not_disable_gpu",
        "//mediapipe:ios",
    ],
)

selects.config_setting_group(
    name = "platform_macos_with_gpu",
    match_all = [
        ":not_disable_gpu",
        "//mediapipe:macos",
    ],
)

selects.config_setting_group(
    name = "platform_apple_with_gpu",
    match_all = [
        ":not_disable_gpu",
        "//mediapipe:apple",
    ],
)

cc_library(
    name = "gpu_buffer",
    srcs = ["gpu_buffer.cc"],
    hdrs = ["gpu_buffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_buffer_format",
        ":gpu_buffer_storage",
        ":gpu_buffer_storage_image_frame",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:logging",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ] + select({
        "//conditions:default": [
            ":gl_texture_buffer",
            ":gl_texture_view",
        ],
        ":platform_ios_with_gpu": [
            ":gl_texture_view",
            ":gpu_buffer_storage_cv_pixel_buffer",
            "//mediapipe/objc:CFHolder",
        ],
        ":platform_macos_with_gpu": [
            ":gl_texture_buffer",
            ":gl_texture_view",
            ":gpu_buffer_storage_cv_pixel_buffer",
            "//mediapipe/objc:CFHolder",
        ],
        ":disable_gpu": [],
    }) + select({
        "//conditions:default": [],
        "//mediapipe:ios": [
            "//mediapipe/objc:util",
        ],
    }),
)

cc_library(
    name = "gpu_buffer_format",
    srcs = ["gpu_buffer_format.cc"],
    hdrs = ["gpu_buffer_format.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/deps:no_destructor",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/port:logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
    ] + select({
        "//conditions:default": [
            ":gl_base",
        ],
        "//mediapipe/gpu:disable_gpu": [],
    }),
)

cc_test(
    name = "gpu_buffer_format_test",
    srcs = ["gpu_buffer_format_test.cc"],
    deps = [
        ":gl_base_hdr",
        ":gpu_buffer_format",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gpu_buffer_storage",
    srcs = ["gpu_buffer_storage.cc"],
    hdrs = ["gpu_buffer_storage.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_buffer_format",
        "//mediapipe/framework/deps:no_destructor",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/tool:type_util",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "gpu_buffer_storage_cv_pixel_buffer",
    srcs = ["gpu_buffer_storage_cv_pixel_buffer.cc"],
    hdrs = ["gpu_buffer_storage_cv_pixel_buffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gl_base",
        ":gl_context",
        ":gl_texture_view",
        ":gpu_buffer_storage",
        ":gpu_buffer_storage_image_frame",
        ":image_frame_view",
        "//mediapipe/objc:CFHolder",
        "//mediapipe/objc:util",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
    ],
)

mediapipe_cc_test(
    name = "gpu_buffer_storage_cv_pixel_buffer_test",
    size = "small",
    timeout = "long",
    srcs = ["gpu_buffer_storage_cv_pixel_buffer_test.cc"],
    platforms = ["ios"],
    deps = [
        ":gl_texture_buffer",
        ":gl_texture_util",
        ":gpu_buffer",
        ":gpu_buffer_storage_cv_pixel_buffer",
        ":gpu_test_base",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/tool:test_util",
        "//mediapipe/objc:util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cv_texture_cache_manager",
    srcs = ["cv_texture_cache_manager.cc"],
    hdrs = ["cv_texture_cache_manager.h"],
    deps = [
        ":pixel_buffer_pool_util",
        "//mediapipe/framework/port:logging",
        "//mediapipe/objc:CFHolder",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "cv_pixel_buffer_pool_wrapper",
    srcs = ["cv_pixel_buffer_pool_wrapper.cc"],
    hdrs = ["cv_pixel_buffer_pool_wrapper.h"],
    copts = select({
        "//conditions:default": [],
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",
        ],
    }),
    deps = [
        ":cv_texture_cache_manager",
        ":gpu_buffer_format",
        ":multi_pool",
        ":pixel_buffer_pool_util",
        "//mediapipe/framework/port:logging",
        "//mediapipe/objc:CFHolder",
        "//mediapipe/objc:util",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "gpu_buffer_storage_image_frame",
    srcs = ["gpu_buffer_storage_image_frame.cc"],
    hdrs = ["gpu_buffer_storage_image_frame.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":frame_buffer_view",
        ":gpu_buffer_format",
        ":gpu_buffer_storage",
        ":image_frame_view",
        "//mediapipe/framework/formats:frame_buffer",
        "//mediapipe/framework/formats:image_frame",
        "@com_google_absl//absl/log:absl_check",
    ],
)

cc_library(
    name = "image_frame_view",
    hdrs = ["image_frame_view.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_buffer_format",
        ":gpu_buffer_storage",
        "//mediapipe/framework/formats:image_frame",
    ],
)

cc_library(
    name = "frame_buffer_view",
    hdrs = ["frame_buffer_view.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_buffer_storage",
        "//mediapipe/framework/formats:frame_buffer",
    ],
)

cc_library(
    name = "gpu_buffer_storage_yuv_image",
    srcs = ["gpu_buffer_storage_yuv_image.cc"],
    hdrs = ["gpu_buffer_storage_yuv_image.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":frame_buffer_view",
        ":gpu_buffer_format",
        ":gpu_buffer_storage",
        ":image_frame_view",
        "//mediapipe/framework/formats:frame_buffer",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:yuv_image",
        "//mediapipe/util/frame_buffer:frame_buffer_util",
        "//third_party/libyuv",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
    ],
)

cc_library(
    name = "gpu_buffer_storage_ahwb",
    srcs = ["gpu_buffer_storage_ahwb.cc"],
    hdrs = ["gpu_buffer_storage_ahwb.h"],
    linkopts = select({
        "//conditions:default": [],
        "//mediapipe:android": [
            "-landroid",
        ],
    }),
    deps = [
        ":gl_base_hdr",
        ":gl_context",
        ":gl_texture_buffer",
        ":gl_texture_view",
        ":gpu_buffer_format",
        ":gpu_buffer_storage",
        ":image_frame_view",
        "//mediapipe/framework:port",
        "//mediapipe/framework/formats:ahwb_view",
        "//mediapipe/framework/formats:hardware_buffer",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:ret_check",
        "//third_party/GL:EGL_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:str_format",
    ],
)

mediapipe_proto_library(
    name = "gpu_origin_proto",
    srcs = ["gpu_origin.proto"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "pixel_buffer_pool_util",
    srcs = ["pixel_buffer_pool_util.cc"],
    hdrs = ["pixel_buffer_pool_util.h"],
    copts = [
        "-x objective-c++",
        "-Wno-shorten-64-to-32",
        "-fobjc-arc",  # enable reference-counting
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/objc:util",
        "//third_party/apple_frameworks:Accelerate",
        "//third_party/apple_frameworks:CoreGraphics",
        "//third_party/apple_frameworks:CoreVideo",
    ],
)

cc_library(
    name = "metal_shared_resources",
    srcs = ["metal_shared_resources.cc"],
    hdrs = ["metal_shared_resources.h"],
    copts = [
        "-x objective-c++",
        "-Wno-shorten-64-to-32",
        "-fobjc-arc",  # enable reference-counting
    ],
    features = ["-layering_check"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/port:ret_check",
        "@google_toolbox_for_mac//:GTM_Defines",
    ] + [
        "//third_party/apple_frameworks:CoreVideo",
        "//third_party/apple_frameworks:Metal",
    ],
    alwayslink = 1,
)

cc_library(
    name = "MPPMetalUtil",
    srcs = ["MPPMetalUtil.cc"],
    hdrs = ["MPPMetalUtil.h"],
    copts = [
        "-x objective-c++",
        "-Wno-shorten-64-to-32",
        "-fobjc-arc",  # enable reference-counting
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/objc:mediapipe_framework_ios",
        "//third_party/apple_frameworks:CoreVideo",
        "//third_party/apple_frameworks:Metal",
        "@com_google_absl//absl/time",
        "@google_toolbox_for_mac//:GTM_Defines",
    ],
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "gl_context_options_proto",
    srcs = ["gl_context_options.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

# This is a hack needed to work around some issues with strict hdrs_check.
# See e.g. b/67524270.
cc_library(
    name = "gpu_shared_data_header",
    textual_hdrs = [
        "gpu_shared_data_internal.h",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":gl_base",
        ":gl_context",
    ],
)

alias(
    name = "gpu_shared_data_internal",
    actual = select({
        "//conditions:default": ":gpu_shared_data_internal_actual",
        ":disable_gpu": ":gpu_shared_data_internal_stub",
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gpu_shared_data_internal_stub",
    visibility = ["//visibility:private"],
    deps = [
        ":gl_context_options_cc_proto",
        ":graph_support",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_node",
        "//mediapipe/framework:executor",
        "//mediapipe/framework:port",
        "//mediapipe/framework/deps:no_destructor",
        "//mediapipe/framework/port:ret_check",
    ],
)

cc_library(
    name = "gpu_shared_data_internal_actual",
    srcs = [
        "gpu_shared_data_internal.cc",
    ],
    hdrs = [
        "gpu_shared_data_internal.h",
    ],
    copts = select({
        "//conditions:default": [],
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
    }),
    visibility = ["//visibility:private"],
    deps = [
        ":gl_base",
        ":gl_context",
        ":gl_context_options_cc_proto",
        ":gpu_buffer_multi_pool",
        ":gpu_shared_data_header",
        ":graph_support",
        ":multi_pool",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_node",
        "//mediapipe/framework:executor",
        "//mediapipe/framework/deps:no_destructor",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ] + select({
        "//conditions:default": [],
        "//mediapipe:apple": [
            ":cv_texture_cache_manager",
            ":metal_shared_resources",
        ],
    }),
)

cc_library(
    name = "gl_texture_buffer_pool",
    srcs = ["gl_texture_buffer_pool.cc"],
    hdrs = ["gl_texture_buffer_pool.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gl_texture_buffer",
        ":multi_pool",
        ":reusable_pool",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "reusable_pool",
    hdrs = ["reusable_pool.h"],
    deps = [
        ":multi_pool",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "multi_pool",
    hdrs = ["multi_pool.h"],
    deps = [
        "//mediapipe/framework/port:status",
        "//mediapipe/util:resource_cache",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "gpu_buffer_multi_pool",
    srcs = ["gpu_buffer_multi_pool.cc"],
    hdrs = ["gpu_buffer_multi_pool.h"] + select({
        "//conditions:default": [
        ],
        "//mediapipe:ios": [
            # The inclusions check does not see that this is provided by
            # pixel_buffer_pool_util, so we include it here too. This is
            # b/28066691.
            "pixel_buffer_pool_util.h",
        ],
        "//mediapipe:macos": [
        ],
    }),
    copts = select({
        "//conditions:default": [],
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":gl_base",
        ":gpu_buffer",
        ":gpu_shared_data_header",
        ":multi_pool",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_node",
        "//mediapipe/framework/port:logging",
        "//mediapipe/util:resource_cache",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ] + select({
        "//conditions:default": [
            ":gl_texture_buffer",
            ":gl_texture_buffer_pool",
        ],
        "//mediapipe:ios": [
            ":cv_pixel_buffer_pool_wrapper",
            ":cv_texture_cache_manager",
            ":pixel_buffer_pool_util",
            "//mediapipe/objc:CFHolder",
            "//mediapipe/objc:util",
        ],
        "//mediapipe:macos": [
            ":cv_pixel_buffer_pool_wrapper",
            ":cv_texture_cache_manager",
            ":gl_texture_buffer",
            ":gl_texture_buffer_pool",
            ":pixel_buffer_pool_util",
        ],
    }),
)

cc_library(
    name = "gl_texture_util",
    srcs = ["gl_texture_util.cc"],
    hdrs = ["gl_texture_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gl_base",
        ":gl_texture_view",
    ],
)

cc_library(
    name = "shader_util",
    srcs = ["shader_util.cc"],
    hdrs = ["shader_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gl_base",
        "//mediapipe/framework/port:logging",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "egl_surface_holder",
    hdrs = ["egl_surface_holder.h"],
    deps = [
        ":gl_base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "gl_calculator_helper",
    srcs = [
        "gl_calculator_helper.cc",
    ],
    hdrs = [
        "gl_calculator_helper.h",
    ],
    linkopts = select({
        "//conditions:default": [],
        "//mediapipe:apple": [
            "-framework AVFoundation",
            "-framework CoreVideo",
            "-framework CoreGraphics",
            "-framework CoreMedia",
            "-framework GLKit",
            "-framework QuartzCore",
        ],
    }) + select({
        "//conditions:default": [],
        "//mediapipe:macos": [
            "-framework AppKit",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":gl_base",
        ":gl_context",
        ":gl_texture_buffer_pool",
        ":gpu_buffer",
        ":gpu_buffer_format",
        ":gpu_buffer_multi_pool",
        ":gpu_service",
        ":gpu_shared_data_internal",
        ":graph_support",
        ":image_frame_view",
        ":shader_util",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_contract",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_node",
        "//mediapipe/framework:demangle",
        "//mediapipe/framework:legacy_calculator_support",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:packet_set",
        "//mediapipe/framework:packet_type",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/deps:registration",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:map_util",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ] + select({
        "//conditions:default": [
        ],
        "//mediapipe:apple": [],
    }),
)

cc_library(
    name = "MPPMetalHelper",
    srcs = ["MPPMetalHelper.cc"],
    hdrs = ["MPPMetalHelper.h"],
    copts = [
        "-Wno-shorten-64-to-32",
        "-x objective-c++",
        "-fobjc-arc",
    ],
    features = ["-layering_check"],
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_service",
        ":gpu_shared_data_internal",
        ":graph_support",
        "//mediapipe/objc:mediapipe_framework_ios",
        "//third_party/apple_frameworks:CoreVideo",
        "//third_party/apple_frameworks:Metal",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@google_toolbox_for_mac//:GTM_Defines",
    ],
)

mediapipe_proto_library(
    name = "scale_mode_proto",
    srcs = ["scale_mode.proto"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gl_quad_renderer",
    srcs = ["gl_quad_renderer.cc"],
    hdrs = ["gl_quad_renderer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gl_base",
        ":gl_simple_shaders",
        ":scale_mode_cc_proto",
        ":shader_util",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "gl_simple_shaders",
    srcs = ["gl_simple_shaders.cc"],
    hdrs = ["gl_simple_shaders.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gl_base",
    ],
)

### General calculator superclasses

cc_library(
    name = "gl_simple_calculator",
    srcs = ["gl_simple_calculator.cc"],
    hdrs = ["gl_simple_calculator.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gl_calculator_helper",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
    ],
)

### Converters

cc_library(
    name = "gpu_buffer_to_image_frame_calculator",
    srcs = ["gpu_buffer_to_image_frame_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":gl_calculator_helper",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ] + select({
        "//conditions:default": [],
        "//mediapipe:apple": ["//mediapipe/objc:util"],
    }),
    alwayslink = 1,
)

cc_library(
    name = "image_frame_to_gpu_buffer_calculator",
    srcs = ["image_frame_to_gpu_buffer_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":gl_calculator_helper",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:status",
    ] + select({
        "//conditions:default": [],
        "//mediapipe:apple": ["//mediapipe/objc:util"],
    }),
    alwayslink = 1,
)

### Simple calculators

mediapipe_proto_library(
    name = "gl_animation_overlay_calculator_proto",
    srcs = ["gl_animation_overlay_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "gl_scaler_calculator_proto",
    srcs = ["gl_scaler_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        ":scale_mode_proto",
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "gl_scaler_calculator",
    srcs = ["gl_scaler_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":gl_calculator_helper",
        ":gl_quad_renderer",
        ":gl_scaler_calculator_cc_proto",
        ":gl_simple_shaders",
        ":gpu_buffer_format",
        ":shader_util",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:options_util",
    ],
    alwayslink = 1,
)

cc_library(
    name = "gl_surface_sink_calculator",
    srcs = ["gl_surface_sink_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":egl_surface_holder",
        ":gl_base_hdr",
        ":gl_calculator_helper",
        ":gl_quad_renderer",
        ":gl_surface_sink_calculator_cc_proto",
        ":gpu_buffer",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "gl_surface_sink_calculator_proto",
    srcs = ["gl_surface_sink_calculator.proto"],
    deps = [
        ":scale_mode_proto",
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

### Metal calculators

metal_library(
    name = "simple_shaders_mtl",
    srcs = ["simple_shaders.metal"],
    hdrs = ["metal_shader_base.h"],
)

# Only needed for cc_library depending on simple_shaders_mtl.
objc_library(
    name = "simple_shaders_for_cc",
    hdrs = ["metal_shader_base.h"],
    deps = [":simple_shaders_mtl"],
)

mediapipe_proto_library(
    name = "copy_calculator_proto",
    srcs = ["copy_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

objc_library(
    name = "metal_copy_calculator",
    srcs = ["MetalCopyCalculator.mm"],
    features = ["-layering_check"],
    visibility = ["//visibility:public"],
    deps = [
        ":MPPMetalHelper",
        ":copy_calculator_cc_proto",
        ":simple_shaders_mtl",
        "//mediapipe/objc:mediapipe_framework_ios",
        "//third_party/apple_frameworks:CoreVideo",
        "//third_party/apple_frameworks:Metal",
    ],
    alwayslink = 1,
)

objc_library(
    name = "metal_rgb_weight_calculator",
    srcs = ["MetalRgbWeightCalculator.mm"],
    features = ["-layering_check"],
    visibility = ["//visibility:public"],
    deps = [
        ":MPPMetalHelper",
        ":simple_shaders_mtl",
        "//mediapipe/objc:mediapipe_framework_ios",
        "//third_party/apple_frameworks:CoreVideo",
        "//third_party/apple_frameworks:Metal",
    ],
    alwayslink = 1,
)

objc_library(
    name = "metal_sobel_calculator",
    srcs = ["MetalSobelCalculator.mm"],
    features = ["-layering_check"],
    visibility = ["//visibility:public"],
    deps = [
        ":MPPMetalHelper",
        ":simple_shaders_mtl",
        "//mediapipe/objc:mediapipe_framework_ios",
        "//third_party/apple_frameworks:CoreVideo",
        "//third_party/apple_frameworks:Metal",
    ],
    alwayslink = 1,
)

objc_library(
    name = "metal_sobel_compute_calculator",
    srcs = ["MetalSobelComputeCalculator.mm"],
    features = ["-layering_check"],
    visibility = ["//visibility:public"],
    deps = [
        ":MPPMetalHelper",
        ":simple_shaders_mtl",
        "//mediapipe/objc:mediapipe_framework_ios",
        "//third_party/apple_frameworks:CoreVideo",
        "//third_party/apple_frameworks:Metal",
    ],
    alwayslink = 1,
)

objc_library(
    name = "mps_sobel_calculator",
    srcs = ["MPSSobelCalculator.mm"],
    copts = ["-std=c++17"],
    features = ["-layering_check"],
    visibility = ["//visibility:public"],
    deps = [
        ":MPPMetalHelper",
        "//mediapipe/objc:mediapipe_framework_ios",
        "//third_party/apple_frameworks:CoreVideo",
        "//third_party/apple_frameworks:Metal",
        "//third_party/apple_frameworks:MetalPerformanceShaders",
    ],
    alwayslink = 1,
)

objc_library(
    name = "mps_threshold_calculator",
    srcs = ["MPSThresholdCalculator.mm"],
    visibility = ["//visibility:public"],
    deps = [
        ":MPPMetalHelper",
        "//mediapipe/objc:mediapipe_framework_ios",
        "//third_party/apple_frameworks:CoreVideo",
        "//third_party/apple_frameworks:Metal",
        "//third_party/apple_frameworks:MetalPerformanceShaders",
    ],
    alwayslink = 1,
)

MIN_IOS_VERSION = "12.0"

test_suite(
    name = "ios",
    tags = ["ios"],
)

test_suite(
    name = "metal",
    tags = ["metal"],
)

objc_library(
    name = "gl_ios_test_lib",
    testonly = 1,
    srcs = [
        "gl_ios_test.mm",
        "metal_shared_resources_test.mm",
    ],
    copts = [
        "-Wno-shorten-64-to-32",
    ],
    data = [
        "//mediapipe/objc:testdata/googlelogo_color_272x92dp.png",
    ],
    features = ["-layering_check"],
    deps = [
        ":gl_scaler_calculator",
        ":gpu_buffer_to_image_frame_calculator",
        ":gpu_shared_data_internal",
        ":image_frame_to_gpu_buffer_calculator",
        ":metal_shared_resources",
        "//mediapipe/framework/port:threadpool",
        "//mediapipe/framework/tool:source",
        "//mediapipe/objc:MPPGraphTestBase",
        "//mediapipe/objc:mediapipe_framework_ios",
        "@com_google_absl//absl/memory",
    ] + select({
        "//mediapipe:ios_i386": [],
        "//mediapipe:ios_x86_64": [],
        "//conditions:default": [
            ":metal_rgb_weight_calculator",
        ],
    }),
)

ios_unit_test(
    name = "gl_ios_test",
    minimum_os_version = MIN_IOS_VERSION,
    runner = "//testing/utp/ios:IOS_LATEST",
    tags = [
        "ios",
    ],
    deps = [":gl_ios_test_lib"],
)

mediapipe_cc_test(
    name = "gpu_buffer_storage_ahwb_test",
    size = "small",
    srcs = ["gpu_buffer_storage_ahwb_test.cc"],
    exclude_platforms = [
        "ios",
        "wasm",
    ],
    requires_full_emulation = True,
    deps = [
        ":gl_texture_buffer",
        ":gl_texture_util",
        ":gpu_buffer_format",
        ":gpu_buffer_storage_ahwb",
        ":gpu_test_base",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/tool:test_util",
    ],
)

mediapipe_cc_test(
    name = "gpu_buffer_storage_yuv_image_test",
    size = "small",
    srcs = ["gpu_buffer_storage_yuv_image_test.cc"],
    exclude_platforms = [
        "ios",
    ],
    deps = [
        ":gpu_buffer",
        ":gpu_buffer_format",
        ":gpu_buffer_storage_yuv_image",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:yuv_image",
        "//mediapipe/framework/port:gtest_main",
        "//third_party/libyuv",
        "@com_google_absl//absl/log:absl_check",
    ],
)
