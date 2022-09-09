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
#

load("@bazel_skylib//lib:selects.bzl", "selects")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")
load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_binary_graph",
)
load("//mediapipe/framework:mediapipe_cc_test.bzl", "mediapipe_cc_test")
load("//mediapipe/framework:encode_binary_proto.bzl", "encode_binary_proto")

licenses(["notice"])

package(default_visibility = ["//visibility:private"])

exports_files(
    glob(["testdata/image_to_tensor/*"]),
    visibility = [
        "//mediapipe/calculators/image:__subpackages__",
    ],
)

selects.config_setting_group(
    name = "compute_shader_unavailable",
    match_any = [
        "//mediapipe/gpu:disable_gpu",
    ],
)

mediapipe_proto_library(
    name = "audio_to_tensor_calculator_proto",
    srcs = ["audio_to_tensor_calculator.proto"],
    visibility = [
        "//mediapipe/framework:mediapipe_internal",
    ],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "audio_to_tensor_calculator",
    srcs = ["audio_to_tensor_calculator.cc"],
    visibility = [
        "//mediapipe/framework:mediapipe_internal",
    ],
    deps = [
        ":audio_to_tensor_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/formats:time_series_header_cc_proto",
        "//mediapipe/util:time_series_util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_audio_tools//audio/dsp:resampler_q",
        "@org_tensorflow//tensorflow/lite/c:common",
    ],
    alwayslink = 1,
)

cc_test(
    name = "audio_to_tensor_calculator_test",
    srcs = ["audio_to_tensor_calculator_test.cc"],
    deps = [
        ":audio_to_tensor_calculator",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/strings",
        "@com_google_audio_tools//audio/dsp:resampler_q",
        "@org_tensorflow//tensorflow/lite/c:common",
    ],
)

mediapipe_proto_library(
    name = "inference_calculator_proto",
    srcs = ["inference_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

# This target defines the "InferenceCalculator" component, which looks for the available concrete
# implementations linked into the current binary and picks the one to use.
# You can depend on :inference_calculator instead if you want to automatically include a default
# set of implementations tailored for the current build configuration.
# If you want to have precise control of which implementations to include (e.g. for strict binary
# size concerns), depend on those implementations directly, and do not depend on
# :inference_calculator.
# In all cases, use "InferenceCalulator" in your graphs.
cc_library(
    name = "inference_calculator_interface",
    srcs = ["inference_calculator.cc"],
    hdrs = ["inference_calculator.h"],
    copts = select({
        # TODO: fix tensor.h not to require this, if possible
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":inference_calculator_options_lib",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/stream_handler:fixed_size_input_stream_handler",
        "//mediapipe/framework/tool:subgraph_expansion",
        "//mediapipe/util/tflite:tflite_model_loader",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/core/api:op_resolver",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "inference_calculator_gl",
    srcs = ["inference_calculator_gl.cc"],
    tags = ["nomac"],  # config problem with cpuinfo via TF
    visibility = ["//visibility:public"],
    deps = [
        ":inference_calculator_cc_proto",
        ":inference_calculator_interface",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/gpu:gl_calculator_helper",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@org_tensorflow//tensorflow/lite/delegates/gpu:gl_delegate",
    ],
    alwayslink = 1,
)

cc_library(
    name = "inference_calculator_gl_advanced",
    srcs = ["inference_calculator_gl_advanced.cc"],
    tags = ["nomac"],
    visibility = ["//visibility:public"],
    deps = [
        ":inference_calculator_interface",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/gpu:gl_calculator_helper",
        "//mediapipe/util/tflite:tflite_gpu_runner",
        "@org_tensorflow//tensorflow/lite:framework_stable",
    ] + select({
        "//conditions:default": [],
        "//mediapipe:android": [
            "//mediapipe/util/android/file/base",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "inference_calculator_metal",
    srcs = ["inference_calculator_metal.cc"],
    copts = [
        "-x objective-c++",
        "-fobjc-arc",  # enable reference-counting
    ],
    linkopts = [
        "-framework CoreVideo",
        "-framework MetalKit",
    ],
    tags = ["ios"],
    visibility = ["//visibility:public"],
    deps = [
        "inference_calculator_interface",
        "//mediapipe/gpu:MPPMetalHelper",
        "//mediapipe/gpu:MPPMetalUtil",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/objc:mediapipe_framework_ios",
        "//mediapipe/util/tflite:config",
        "@com_google_absl//absl/memory",
        "@org_tensorflow//tensorflow/lite/delegates/gpu:metal_delegate",
        "@org_tensorflow//tensorflow/lite/delegates/gpu:metal_delegate_internal",
        "@org_tensorflow//tensorflow/lite/delegates/gpu/common:shape",
        "@org_tensorflow//tensorflow/lite/delegates/gpu/metal:buffer_convert",
    ],
    alwayslink = 1,
)

cc_library(
    name = "inference_runner",
    hdrs = ["inference_runner.h"],
    copts = select({
        # TODO: fix tensor.h not to require this, if possible
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/formats:tensor",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "inference_interpreter_delegate_runner",
    srcs = ["inference_interpreter_delegate_runner.cc"],
    hdrs = ["inference_interpreter_delegate_runner.h"],
    copts = select({
        # TODO: fix tensor.h not to require this, if possible
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":inference_runner",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/util/tflite:tflite_model_loader",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@org_tensorflow//tensorflow/lite:framework_stable",
        "@org_tensorflow//tensorflow/lite/core/api:op_resolver",
    ],
)

cc_library(
    name = "inference_calculator_cpu",
    srcs = [
        "inference_calculator_cpu.cc",
    ],
    copts = select({
        # TODO: fix tensor.h not to require this, if possible
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":inference_calculator_interface",
        ":inference_calculator_utils",
        ":inference_interpreter_delegate_runner",
        ":inference_runner",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@org_tensorflow//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
        "@org_tensorflow//tensorflow/lite:framework_stable",
        "@org_tensorflow//tensorflow/lite/c:c_api_types",
    ] + select({
        "//conditions:default": [],
        "//mediapipe:android": ["@org_tensorflow//tensorflow/lite/delegates/nnapi:nnapi_delegate"],
    }),
    alwayslink = 1,
)

cc_library(
    name = "inference_calculator_utils",
    srcs = ["inference_calculator_utils.cc"],
    hdrs = ["inference_calculator_utils.h"],
    deps = [
        ":inference_calculator_cc_proto",
        "//mediapipe/framework:port",
    ] + select({
        "//conditions:default": [
            "//mediapipe/util:cpu_util",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "inference_calculator_xnnpack",
    srcs = [
        "inference_calculator_xnnpack.cc",
    ],
    copts = select({
        # TODO: fix tensor.h not to require this, if possible
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":inference_calculator_interface",
        ":inference_calculator_utils",
        ":inference_interpreter_delegate_runner",
        ":inference_runner",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@org_tensorflow//tensorflow/lite:framework_stable",
        "@org_tensorflow//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
    ],
    alwayslink = 1,
)

cc_library(
    name = "inference_calculator_gl_if_compute_shader_available",
    visibility = ["//visibility:public"],
    deps = selects.with_or({
        ":compute_shader_unavailable": [],
        "//conditions:default": [
            ":inference_calculator_gl",
            ":inference_calculator_gl_advanced",
        ],
    }),
)

cc_library(
    name = "inference_calculator",
    visibility = ["//visibility:public"],
    deps = [
        ":inference_calculator_interface",
        ":inference_calculator_cpu",
    ] + select({
        "//conditions:default": [":inference_calculator_gl_if_compute_shader_available"],
        "//mediapipe:ios": [":inference_calculator_metal"],
    }),
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "tensor_converter_calculator_proto",
    srcs = ["tensor_converter_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "tensor_converter_calculator",
    srcs = ["tensor_converter_calculator.cc"],
    copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    features = ["-layering_check"],  # allow depending on tensor_converter_calculator_gpu_deps
    linkopts = select({
        "//mediapipe:apple": [
            "-framework CoreVideo",
            "-framework MetalKit",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":tensor_converter_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework:port",
        "//mediapipe/util:resource_util",
    ] + select({
        "//mediapipe/gpu:disable_gpu": [],
        "//conditions:default": ["tensor_converter_calculator_gpu_deps"],
    }),
    alwayslink = 1,
)

cc_library(
    name = "tensor_converter_calculator_gpu_deps",
    deps = select({
        "//mediapipe:android": [
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gpu_buffer",
        ],
        "//mediapipe:ios": [
            "//mediapipe/gpu:MPPMetalUtil",
            "//mediapipe/gpu:MPPMetalHelper",
            "//mediapipe/objc:mediapipe_framework_ios",
        ],
        "//mediapipe:macos": [],
        "//conditions:default": [
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gl_simple_shaders",
            "//mediapipe/gpu:shader_util",
            "//mediapipe/gpu:gpu_buffer",
        ],
    }),
)

cc_test(
    name = "tensor_converter_calculator_test",
    srcs = ["tensor_converter_calculator_test.cc"],
    deps = [
        ":tensor_converter_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/tool:validate_type",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

mediapipe_proto_library(
    name = "tensors_to_detections_calculator_proto",
    srcs = ["tensors_to_detections_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "tensors_to_detections_calculator",
    srcs = ["tensors_to_detections_calculator.cc"],
    copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    features = ["-layering_check"],  # allow depending on tensors_to_detections_calculator_gpu_deps
    linkopts = select({
        "//mediapipe:apple": [
            "-framework CoreVideo",
            "-framework MetalKit",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":tensors_to_detections_calculator_cc_proto",
        "//mediapipe/framework/formats:detection_cc_proto",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:port",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:location",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/formats/object_detection:anchor_cc_proto",
        "//mediapipe/framework/port:ret_check",
    ] + selects.with_or({
        ":compute_shader_unavailable": [],
        "//conditions:default": [":tensors_to_detections_calculator_gpu_deps"],
    }),
    alwayslink = 1,
)

cc_library(
    name = "tensors_to_detections_calculator_gpu_deps",
    deps = select({
        "//mediapipe:ios": [
            "//mediapipe/gpu:MPPMetalUtil",
            "//mediapipe/gpu:MPPMetalHelper",
        ],
        "//mediapipe:macos": [],
        "//conditions:default": [
            "//mediapipe/gpu:gl_calculator_helper",
        ],
    }),
)

mediapipe_proto_library(
    name = "tensors_to_landmarks_calculator_proto",
    srcs = ["tensors_to_landmarks_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "tensors_to_landmarks_calculator",
    srcs = ["tensors_to_landmarks_calculator.cc"],
    copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":tensors_to_landmarks_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:ret_check",
    ],
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "landmarks_to_tensor_calculator_proto",
    srcs = ["landmarks_to_tensor_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "landmarks_to_tensor_calculator",
    srcs = ["landmarks_to_tensor_calculator.cc"],
    hdrs = ["landmarks_to_tensor_calculator.h"],
    copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":landmarks_to_tensor_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:ret_check",
    ],
    alwayslink = 1,
)

cc_test(
    name = "landmarks_to_tensor_calculator_test",
    srcs = ["landmarks_to_tensor_calculator_test.cc"],
    deps = [
        ":landmarks_to_tensor_calculator",
        ":landmarks_to_tensor_calculator_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

mediapipe_proto_library(
    name = "tensors_to_floats_calculator_proto",
    srcs = ["tensors_to_floats_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "tensors_to_floats_calculator",
    srcs = ["tensors_to_floats_calculator.cc"],
    copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":tensors_to_floats_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:ret_check",
    ],
    alwayslink = 1,
)

cc_test(
    name = "tensors_to_floats_calculator_test",
    srcs = ["tensors_to_floats_calculator_test.cc"],
    deps = [
        ":tensors_to_floats_calculator",
        ":tensors_to_floats_calculator_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tensors_to_classification_calculator",
    srcs = ["tensors_to_classification_calculator.cc"],
    copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":tensors_to_classification_calculator_cc_proto",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:location",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/util:label_map_cc_proto",
        "//mediapipe/util:resource_util",
    ] + select({
        "//mediapipe:android": [
            "//mediapipe/util/android/file/base",
        ],
        "//mediapipe:ios": [
            "//mediapipe/util/android/file/base",
        ],
        "//conditions:default": [
            "//mediapipe/framework/port:file_helpers",
        ],
    }),
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "tensors_to_classification_calculator_proto",
    srcs = ["tensors_to_classification_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util:label_map_proto",
    ],
)

cc_test(
    name = "tensors_to_classification_calculator_test",
    srcs = ["tensors_to_classification_calculator_test.cc"],
    data = ["testdata/labelmap.txt"],
    deps = [
        ":tensors_to_classification_calculator",
        ":tensors_to_classification_calculator_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "image_to_tensor_calculator",
    srcs = ["image_to_tensor_calculator.cc"],
    copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    features = ["-layering_check"],  # allow depending on image_to_tensor_calculator_gpu_deps
    visibility = ["//visibility:public"],
    deps = [
        ":image_to_tensor_calculator_cc_proto",
        ":image_to_tensor_converter",
        ":image_to_tensor_utils",
        ":loose_headers",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:port",
        "//mediapipe/gpu:gpu_origin_cc_proto",
    ] + select({
        "//mediapipe/gpu:disable_gpu": [],
        "//conditions:default": [":image_to_tensor_calculator_gpu_deps"],
    }) + select({
        "//mediapipe/framework/port:disable_opencv": [],
        "//conditions:default": [":image_to_tensor_converter_opencv"],
    }),
    alwayslink = 1,
)

cc_library(
    name = "image_to_tensor_calculator_gpu_deps",
    deps = selects.with_or({
        "//mediapipe:android": [
            ":image_to_tensor_converter_gl_buffer",
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gpu_buffer",
        ],
        "//mediapipe:apple": [
            ":image_to_tensor_converter_metal",
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:MPPMetalHelper",
            "//mediapipe/gpu:gpu_buffer",
        ],
        "//conditions:default": [
            ":image_to_tensor_converter_gl_buffer",
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gpu_buffer",
        ],
    }),
)

mediapipe_proto_library(
    name = "image_to_tensor_calculator_proto",
    srcs = ["image_to_tensor_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/gpu:gpu_origin_proto",
    ],
)

cc_test(
    name = "image_to_tensor_calculator_test",
    srcs = ["image_to_tensor_calculator_test.cc"],
    data = [
        "testdata/image_to_tensor/input.jpg",
        "testdata/image_to_tensor/large_sub_rect.png",
        "testdata/image_to_tensor/large_sub_rect_border_zero.png",
        "testdata/image_to_tensor/large_sub_rect_keep_aspect.png",
        "testdata/image_to_tensor/large_sub_rect_keep_aspect_border_zero.png",
        "testdata/image_to_tensor/large_sub_rect_keep_aspect_with_rotation.png",
        "testdata/image_to_tensor/large_sub_rect_keep_aspect_with_rotation_border_zero.png",
        "testdata/image_to_tensor/medium_sub_rect_keep_aspect.png",
        "testdata/image_to_tensor/medium_sub_rect_keep_aspect_border_zero.png",
        "testdata/image_to_tensor/medium_sub_rect_keep_aspect_with_rotation.png",
        "testdata/image_to_tensor/medium_sub_rect_keep_aspect_with_rotation_border_zero.png",
        "testdata/image_to_tensor/medium_sub_rect_with_rotation.png",
        "testdata/image_to_tensor/medium_sub_rect_with_rotation_border_zero.png",
        "testdata/image_to_tensor/noop_except_range.png",
    ],
    deps = [
        ":image_to_tensor_calculator",
        ":image_to_tensor_converter",
        ":image_to_tensor_utils",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "image_to_tensor_converter",
    hdrs = ["image_to_tensor_converter.h"],
    copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":image_to_tensor_utils",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:statusor",
    ],
)

cc_library(
    name = "image_to_tensor_converter_opencv",
    srcs = ["image_to_tensor_converter_opencv.cc"],
    hdrs = ["image_to_tensor_converter_opencv.h"],
    copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":image_to_tensor_converter",
        ":image_to_tensor_utils",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_opencv",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
    ],
)

cc_library(
    name = "image_to_tensor_converter_gl_buffer",
    srcs = ["image_to_tensor_converter_gl_buffer.cc"],
    hdrs = ["image_to_tensor_converter_gl_buffer.h"],
    deps = ["//mediapipe/framework:port"] + selects.with_or({
        "//mediapipe:apple": [],
        "//conditions:default": [
            ":image_to_tensor_converter",
            ":image_to_tensor_converter_gl_utils",
            ":image_to_tensor_utils",
            "@com_google_absl//absl/strings",
            "//mediapipe/framework:calculator_framework",
            "//mediapipe/framework/formats:rect_cc_proto",
            "//mediapipe/framework/formats:tensor",
            "//mediapipe/framework/port:ret_check",
            "//mediapipe/framework/port:status",
            "//mediapipe/framework/port:statusor",
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/framework/formats:image",
            "//mediapipe/gpu:gpu_buffer_format",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/common:shape",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/common:types",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:command_queue",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_buffer",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_call",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_texture",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:request_gpu_info",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:variable",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl/converters:util",
        ],
    }),
)

cc_library(
    name = "image_to_tensor_converter_gl_texture",
    srcs = ["image_to_tensor_converter_gl_texture.cc"],
    hdrs = ["image_to_tensor_converter_gl_texture.h"],
    deps = ["//mediapipe/framework:port"] + select({
        "//mediapipe/gpu:disable_gpu": [],
        "//conditions:default": [
            ":image_to_tensor_converter",
            ":image_to_tensor_converter_gl_utils",
            ":image_to_tensor_utils",
            "@com_google_absl//absl/strings",
            "//mediapipe/framework:calculator_framework",
            "//mediapipe/framework/formats:tensor",
            "//mediapipe/framework/port:ret_check",
            "//mediapipe/framework/port:status",
            "//mediapipe/framework/port:statusor",
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gl_simple_shaders",
            "//mediapipe/framework/formats:image",
            "//mediapipe/gpu:shader_util",
        ],
    }),
)

cc_library(
    name = "image_to_tensor_converter_gl_utils",
    srcs = ["image_to_tensor_converter_gl_utils.cc"],
    hdrs = ["image_to_tensor_converter_gl_utils.h"],
    deps = ["//mediapipe/framework:port"] + select({
        "//mediapipe/gpu:disable_gpu": [],
        "//conditions:default": [
            "//mediapipe/gpu:gl_base",
            "//mediapipe/gpu:gl_context",
            "//mediapipe/framework/port:status",
            "//mediapipe/framework/port:statusor",
        ],
    }),
)

cc_library(
    name = "image_to_tensor_converter_metal",
    srcs = ["image_to_tensor_converter_metal.cc"],
    hdrs = ["image_to_tensor_converter_metal.h"],
    copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    linkopts = select({
        "//mediapipe:apple": [
            "-framework CoreVideo",
            "-framework MetalKit",
        ],
        "//conditions:default": [],
    }),
    deps = ["//mediapipe/framework:port"] + select({
        "//mediapipe:apple": [
            ":image_to_tensor_converter",
            ":image_to_tensor_utils",
            "//mediapipe/gpu:MPPMetalHelper",
            "@com_google_absl//absl/strings",
            "//mediapipe/framework:calculator_framework",
            "//mediapipe/framework/formats:rect_cc_proto",
            "//mediapipe/framework/formats:tensor",
            "//mediapipe/framework/port:ret_check",
            "//mediapipe/framework/port:status",
            "//mediapipe/framework/port:statusor",
            "//mediapipe/framework/formats:image",
            "//mediapipe/gpu:gpu_buffer_format",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/common:shape",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/common:types",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "image_to_tensor_utils",
    srcs = ["image_to_tensor_utils.cc"],
    hdrs = ["image_to_tensor_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:statusor",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_test(
    name = "image_to_tensor_utils_test",
    srcs = ["image_to_tensor_utils_test.cc"],
    deps = [
        ":image_to_tensor_utils",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:gtest_main",
    ],
)

# Copied from /mediapipe/calculators/tflite/BUILD
selects.config_setting_group(
    name = "gpu_inference_disabled",
    match_any = [
        "//mediapipe/gpu:disable_gpu",
    ],
)

mediapipe_proto_library(
    name = "tensors_to_segmentation_calculator_proto",
    srcs = ["tensors_to_segmentation_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/gpu:gpu_origin_proto",
    ],
)

cc_library(
    name = "tensors_to_segmentation_calculator",
    srcs = ["tensors_to_segmentation_calculator.cc"],
    copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":tensors_to_segmentation_calculator_cc_proto",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:port",
        "//mediapipe/util:resource_util",
        "@org_tensorflow//tensorflow/lite:framework",
        "//mediapipe/gpu:gpu_origin_cc_proto",
        "//mediapipe/framework/port:statusor",
    ] + selects.with_or({
        "//mediapipe/gpu:disable_gpu": [],
        "//conditions:default": [
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gl_simple_shaders",
            "//mediapipe/gpu:gpu_buffer",
            "//mediapipe/gpu:shader_util",
        ],
    }) + selects.with_or({
        ":gpu_inference_disabled": [],
        "//mediapipe:ios": [
            "//mediapipe/gpu:MPPMetalUtil",
            "//mediapipe/gpu:MPPMetalHelper",
        ],
        "//conditions:default": [
            "@org_tensorflow//tensorflow/lite/delegates/gpu:gl_delegate",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_program",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_shader",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_texture",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl/converters:util",
        ],
    }) + select({
        "//mediapipe/framework/port:disable_opencv": [],
        "//conditions:default": [
            "//mediapipe/framework/formats:image_opencv",
            "//mediapipe/framework/port:opencv_imgproc",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "tensors_dequantization_calculator",
    srcs = ["tensors_dequantization_calculator.cc"],
    copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

# For a more maintainable build this target should not exist and the headers
# should  be split into the existing cc_library targets, but this change was
# automatically  done so that we can remove long standing issues and complexity
# in the build system. It's up to the OWNERS of this package to get rid of it or
# not. The use of the textual_hdrs attribute is discouraged, use hdrs instead.
# Here it is used to avoid header parsing errors in packages where the feature
# parse_headers was enabled since loose headers were not being parsed.
cc_library(
    name = "loose_headers",
    tags = ["avoid_dep"],
    textual_hdrs = [
        "image_to_tensor_converter_gl_buffer.h",
        "image_to_tensor_converter_gl_texture.h",
    ],
    visibility = [":__pkg__"],
)

cc_test(
    name = "tensors_dequantization_calculator_test",
    srcs = ["tensors_dequantization_calculator_test.cc"],
    deps = [
        ":tensors_dequantization_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/status",
    ],
)
