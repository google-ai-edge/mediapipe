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

load("//mediapipe/gpu:metal.bzl", "metal_library")
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_unit_test")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library")

licenses(["notice"])  # Apache 2.0

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
    deps = ["//mediapipe/framework:graph_service"],
)

cc_library(
    name = "graph_support",
    hdrs = ["graph_support.h"],
    visibility = ["//visibility:public"],
    deps = [":gpu_service"],
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
    "//mediapipe:ios": [],
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
            "-fobjc-arc",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":gl_base",
        ":gl_thread_collector",
        "//mediapipe/framework:executor",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "//mediapipe/framework/port:threadpool",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
        "//mediapipe/framework:mediapipe_profiling",
        "//mediapipe/framework:timestamp",
    ] + select({
        "//conditions:default": [],
        "//mediapipe:apple": [
            "//mediapipe/objc:CFHolder",
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
        ":gpu_buffer_format",
        # TODO: remove this dependency. Some other teams' tests
        # depend on having an indirect image_frame dependency, need to be
        # fixed first.
        "//mediapipe/framework/formats:image_frame",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "gpu_buffer",
    hdrs = ["gpu_buffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gl_base",
        ":gpu_buffer_format",
    ] + select({
        "//conditions:default": [
            ":gl_texture_buffer",
        ],
        "//mediapipe:ios": [
            "//mediapipe/objc:CFHolder",
        ],
        "//mediapipe:macos": [
            "//mediapipe/objc:CFHolder",
            ":gl_texture_buffer",
        ],
    }),
)

cc_library(
    name = "gpu_buffer_format",
    srcs = ["gpu_buffer_format.cc"],
    hdrs = ["gpu_buffer_format.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gl_base",
        "//mediapipe/framework/deps:no_destructor",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/port:logging",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

objc_library(
    name = "pixel_buffer_pool_util",
    srcs = ["pixel_buffer_pool_util.mm"],
    hdrs = ["pixel_buffer_pool_util.h"],
    copts = [
        "-Wno-shorten-64-to-32",
    ],
    sdk_frameworks = [
        "Accelerate",
        "CoreGraphics",
        "CoreVideo",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/objc:util",
    ],
)

objc_library(
    name = "MPPGraphGPUData",
    srcs = [
        "MPPGraphGPUData.mm",
        "gpu_shared_data_internal.cc",
    ],
    hdrs = ["MPPGraphGPUData.h"],
    copts = [
        "-x objective-c++",
        "-Wno-shorten-64-to-32",
    ],
    sdk_frameworks = [
        "CoreVideo",
        "Metal",
    ] + select({
        "//conditions:default": [
            "OpenGLES",
        ],
        "//mediapipe:macos": [
            "OpenGL",
            "AppKit",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":gl_base",
        ":gl_context",
        ":gpu_buffer_multi_pool",
        ":gpu_shared_data_header",
        ":graph_support",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/gpu:gl_context_options_cc_proto",
        "@google_toolbox_for_mac//:GTM_Defines",
    ],
)

objc_library(
    name = "MPPMetalUtil",
    srcs = ["MPPMetalUtil.mm"],
    hdrs = ["MPPMetalUtil.h"],
    copts = [
        "-x objective-c++",
        "-Wno-shorten-64-to-32",
    ],
    sdk_frameworks = [
        "CoreVideo",
        "Metal",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/objc:mediapipe_framework_ios",
        "@com_google_absl//absl/time",
        "@google_toolbox_for_mac//:GTM_Defines",
    ],
)

proto_library(
    name = "gl_context_options_proto",
    srcs = ["gl_context_options.proto"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework:calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "gl_context_options_cc_proto",
    srcs = ["gl_context_options.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":gl_context_options_proto"],
)

# This is a hack needed to work around some issues with strict hdrs_check.
# See e.g. b/67524270.
cc_library(
    name = "gpu_shared_data_header",
    textual_hdrs = [
        "gpu_shared_data_internal.h",
    ] + select({
        "//conditions:default": [],
        "//mediapipe:apple": [
            "MPPGraphGPUData.h",
        ],
    }),
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
    hdrs = [
        "gpu_shared_data_internal.h",
    ],
    defines = ["MEDIAPIPE_DISABLE_GPU"],
    visibility = ["//visibility:private"],
    deps = [
        ":graph_support",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_node",
        "//mediapipe/framework:executor",
        "//mediapipe/framework/deps:no_destructor",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/gpu:gl_context_options_cc_proto",
    ],
)

cc_library(
    name = "gpu_shared_data_internal_actual",
    srcs = select({
        "//conditions:default": [
            "gpu_shared_data_internal.cc",
        ],
        # iOS uses an Objective-C++ version of this, built in MPPGraphGPUData.
        "//mediapipe:apple": [],
    }),
    hdrs = [
        "gpu_shared_data_internal.h",
    ],
    visibility = ["//visibility:private"],
    deps = [
        "//mediapipe/gpu:gl_context_options_cc_proto",
        ":graph_support",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:executor",
        "//mediapipe/framework:calculator_node",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/deps:no_destructor",
        ":gl_base",
        ":gl_context",
        ":gpu_buffer_multi_pool",
        ":gpu_shared_data_header",
    ] + select({
        "//conditions:default": [],
        "//mediapipe:apple": [
            ":MPPGraphGPUData",
        ],
    }),
)

cc_library(
    name = "gpu_buffer_multi_pool",
    srcs = ["gpu_buffer_multi_pool.cc"] + select({
        "//conditions:default": [
            "gl_texture_buffer_pool.cc",
        ],
        "//mediapipe:ios": [],
        "//mediapipe:macos": [
            "gl_texture_buffer_pool.cc",
        ],
    }),
    hdrs = ["gpu_buffer_multi_pool.h"] + select({
        "//conditions:default": [
            "gl_texture_buffer_pool.h",
        ],
        "//mediapipe:ios": [
            # The inclusions check does not see that this is provided by
            # pixel_buffer_pool_util, so we include it here too. This is
            # b/28066691.
            "pixel_buffer_pool_util.h",
        ],
        "//mediapipe:macos": [
            "gl_texture_buffer_pool.h",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":gl_base",
        ":gpu_buffer",
        ":gpu_shared_data_header",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_node",
        "//mediapipe/framework/port:logging",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
    ] + select({
        "//conditions:default": [
            ":gl_texture_buffer",
        ],
        "//mediapipe:ios": [
            ":pixel_buffer_pool_util",
            "//mediapipe/objc:CFHolder",
        ],
        "//mediapipe:macos": [
            ":pixel_buffer_pool_util",
            ":gl_texture_buffer",
        ],
    }),
)

cc_library(
    name = "shader_util",
    srcs = ["shader_util.cc"],
    hdrs = ["shader_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gl_base",
        "//mediapipe/framework/port:logging",
    ],
)

HELPER_ANDROID_SRCS = [
    "gl_calculator_helper_impl_android.cc",
    "gl_calculator_helper_impl_common.cc",
]

HELPER_ANDROID_HDRS = [
    "egl_surface_holder.h",
]

HELPER_COMMON_SRCS = [
    "gl_calculator_helper.cc",
]

HELPER_COMMON_HDRS = [
    "gl_calculator_helper.h",
    "gl_calculator_helper_impl.h",
]

HELPER_IOS_SRCS = [
    "gl_calculator_helper_impl_ios.mm",
    "gl_calculator_helper_impl_common.cc",
]

HELPER_IOS_FRAMEWORKS = [
    "AVFoundation",
    "CoreVideo",
    "CoreGraphics",
    "CoreMedia",
    "GLKit",
    "QuartzCore",
] + select({
    "//conditions:default": [
        "OpenGLES",
    ],
    "//mediapipe:macos": [
        "OpenGL",
        "AppKit",
    ],
})

cc_library(
    name = "gl_calculator_helper",
    srcs = select({
        "//conditions:default": HELPER_COMMON_SRCS + HELPER_ANDROID_SRCS,
        "//mediapipe:apple": [],
    }),
    hdrs = HELPER_COMMON_HDRS + select({
        "//conditions:default": HELPER_ANDROID_HDRS,
        "//mediapipe:apple": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":gl_base",
        ":gl_context",
        ":gpu_buffer",
        ":gpu_buffer_format",
        ":gpu_buffer_multi_pool",
        ":gpu_shared_data_internal",
        ":gpu_service",
        ":graph_support",
        ":shader_util",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_node",
        "//mediapipe/framework:calculator_contract",
        "//mediapipe/framework:demangle",
        "//mediapipe/framework:legacy_calculator_support",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:packet_set",
        "//mediapipe/framework:packet_type",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
        "//mediapipe/framework/deps:registration",
        "//mediapipe/framework/port:map_util",
    ] + select({
        "//conditions:default": [
        ],
        "//mediapipe:apple": [
            ":gl_calculator_helper_ios",
            "//mediapipe/objc:util",
        ],
    }),
)

objc_library(
    name = "gl_calculator_helper_ios",
    srcs = HELPER_COMMON_SRCS + HELPER_IOS_SRCS,
    hdrs = HELPER_COMMON_HDRS,
    copts = [
        "-Wno-shorten-64-to-32",
    ],
    sdk_frameworks = HELPER_IOS_FRAMEWORKS,
    visibility = ["//visibility:public"],
    deps = [
        ":gl_base",
        ":gl_context",
        ":gpu_buffer",
        ":gpu_buffer_multi_pool",
        ":gpu_service",
        ":gpu_shared_data_internal",
        ":shader_util",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/objc:mediapipe_framework_ios",
        "//mediapipe/objc:util",
    ],
)

objc_library(
    name = "MPPMetalHelper",
    srcs = ["MPPMetalHelper.mm"],
    hdrs = ["MPPMetalHelper.h"],
    copts = [
        "-Wno-shorten-64-to-32",
    ],
    sdk_frameworks = [
        "CoreVideo",
        "Metal",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_shared_data_internal",
        ":graph_support",
        "//mediapipe/objc:mediapipe_framework_ios",
        "@google_toolbox_for_mac//:GTM_Defines",
    ],
)

proto_library(
    name = "scale_mode_proto",
    srcs = ["scale_mode.proto"],
    visibility = ["//visibility:public"],
)

mediapipe_cc_proto_library(
    name = "scale_mode_cc_proto",
    srcs = ["scale_mode.proto"],
    visibility = ["//visibility:public"],
    deps = [":scale_mode_proto"],
)

cc_library(
    name = "gl_quad_renderer",
    srcs = ["gl_quad_renderer.cc"],
    hdrs = ["gl_quad_renderer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gl_base",
        ":gl_simple_shaders",
        ":shader_util",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:scale_mode_cc_proto",
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
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

proto_library(
    name = "gl_scaler_calculator_proto",
    srcs = ["gl_scaler_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/gpu:scale_mode_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "gl_scaler_calculator_cc_proto",
    srcs = ["gl_scaler_calculator.proto"],
    cc_deps = [
        ":scale_mode_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":gl_scaler_calculator_proto"],
)

cc_library(
    name = "gl_scaler_calculator",
    srcs = ["gl_scaler_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":gl_calculator_helper",
        ":gl_quad_renderer",
        ":gl_simple_shaders",
        ":shader_util",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:options_util",
        "//mediapipe/gpu:gl_scaler_calculator_cc_proto",
    ],
    alwayslink = 1,
)

cc_library(
    name = "gl_surface_sink_calculator",
    srcs = ["gl_surface_sink_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":gl_calculator_helper",
        ":gl_quad_renderer",
        ":shader_util",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:gl_surface_sink_calculator_cc_proto",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = 1,
)

proto_library(
    name = "gl_surface_sink_calculator_proto",
    srcs = ["gl_surface_sink_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/gpu:scale_mode_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "gl_surface_sink_calculator_cc_proto",
    srcs = ["gl_surface_sink_calculator.proto"],
    cc_deps = [
        ":scale_mode_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":gl_surface_sink_calculator_proto"],
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

proto_library(
    name = "copy_calculator_proto",
    srcs = ["copy_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework:calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "copy_calculator_cc_proto",
    srcs = ["copy_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":copy_calculator_proto"],
)

objc_library(
    name = "metal_copy_calculator",
    srcs = ["MetalCopyCalculator.mm"],
    sdk_frameworks = [
        "CoreVideo",
        "Metal",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":MPPMetalHelper",
        ":simple_shaders_mtl",
        "//mediapipe/gpu:copy_calculator_cc_proto",
        "//mediapipe/objc:mediapipe_framework_ios",
    ],
    alwayslink = 1,
)

objc_library(
    name = "metal_rgb_weight_calculator",
    srcs = ["MetalRgbWeightCalculator.mm"],
    sdk_frameworks = [
        "CoreVideo",
        "Metal",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":MPPMetalHelper",
        ":simple_shaders_mtl",
        "//mediapipe/objc:mediapipe_framework_ios",
    ],
    alwayslink = 1,
)

objc_library(
    name = "metal_sobel_calculator",
    srcs = ["MetalSobelCalculator.mm"],
    sdk_frameworks = [
        "CoreVideo",
        "Metal",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":MPPMetalHelper",
        ":simple_shaders_mtl",
        "//mediapipe/objc:mediapipe_framework_ios",
    ],
    alwayslink = 1,
)

objc_library(
    name = "metal_sobel_compute_calculator",
    srcs = ["MetalSobelComputeCalculator.mm"],
    sdk_frameworks = [
        "CoreVideo",
        "Metal",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":MPPMetalHelper",
        ":simple_shaders_mtl",
        "//mediapipe/objc:mediapipe_framework_ios",
    ],
    alwayslink = 1,
)

objc_library(
    name = "mps_sobel_calculator",
    srcs = ["MPSSobelCalculator.mm"],
    sdk_frameworks = [
        "CoreVideo",
        "Metal",
        "MetalPerformanceShaders",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":MPPMetalHelper",
        "//mediapipe/objc:mediapipe_framework_ios",
    ],
    alwayslink = 1,
)

MIN_IOS_VERSION = "9.0"  # For thread_local.

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
        "MPPGraphGPUDataTests.mm",
        "gl_ios_test.mm",
    ],
    copts = [
        "-Wno-shorten-64-to-32",
    ],
    data = [
        "//mediapipe/objc:testdata/googlelogo_color_272x92dp.png",
    ],
    deps = [
        ":MPPGraphGPUData",
        ":gl_scaler_calculator",
        ":gpu_buffer_to_image_frame_calculator",
        ":gpu_shared_data_internal",
        ":image_frame_to_gpu_buffer_calculator",
        "//mediapipe/objc:MPPGraphTestBase",
        "//mediapipe/objc:mediapipe_framework_ios",
        "//mediapipe/framework/tool:source",
        "//mediapipe/framework/port:threadpool",
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
    runner = "//googlemac/iPhone/Shared/Testing/EarlGrey/Runner:IOS_LATEST",
    tags = [
        "ios",
    ],
    deps = [":gl_ios_test_lib"],
)
