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
load("@org_tensorflow//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_proto_library(
    name = "ssd_anchors_calculator_proto",
    srcs = ["ssd_anchors_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/framework/formats/object_detection:anchor_proto",
    ],
)

mediapipe_proto_library(
    name = "tflite_custom_op_resolver_calculator_proto",
    srcs = ["tflite_custom_op_resolver_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "tflite_inference_calculator_proto",
    srcs = ["tflite_inference_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "tflite_converter_calculator_proto",
    srcs = ["tflite_converter_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "tflite_tensors_to_segmentation_calculator_proto",
    srcs = ["tflite_tensors_to_segmentation_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "tflite_tensors_to_detections_calculator_proto",
    srcs = ["tflite_tensors_to_detections_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "tflite_tensors_to_classification_calculator_proto",
    srcs = ["tflite_tensors_to_classification_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "tflite_tensors_to_landmarks_calculator_proto",
    srcs = ["tflite_tensors_to_landmarks_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "ssd_anchors_calculator",
    srcs = ["ssd_anchors_calculator.cc"],
    deps = [
        ":ssd_anchors_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats/object_detection:anchor_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tflite_custom_op_resolver_calculator",
    srcs = ["tflite_custom_op_resolver_calculator.cc"],
    deps = [
        ":tflite_custom_op_resolver_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/util/tflite:cpu_op_resolver",
        "//mediapipe/util/tflite:op_resolver",
        "@org_tensorflow//tensorflow/lite/core/api:op_resolver",
    ],
    alwayslink = 1,
)

filegroup(
    name = "anchor_golden_files",
    srcs = [
        "testdata/anchor_golden_file_0.txt",
        "testdata/anchor_golden_file_1.txt",
        "testdata/anchor_golden_file_2.txt",
    ],
)

cc_test(
    name = "ssd_anchors_calculator_test",
    srcs = ["ssd_anchors_calculator_test.cc"],
    data = [":anchor_golden_files"],
    deps = [
        ":ssd_anchors_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats/object_detection:anchor_cc_proto",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/flags:flag",
    ],
)

selects.config_setting_group(
    name = "gpu_inference_disabled",
    match_any = [
        "//mediapipe/gpu:disable_gpu",
    ],
)

config_setting(
    name = "edge_tpu_usb",
    define_values = {
        "MEDIAPIPE_EDGE_TPU": "usb",
    },
)

config_setting(
    name = "edge_tpu_pci",
    define_values = {
        "MEDIAPIPE_EDGE_TPU": "pci",
    },
)

config_setting(
    name = "edge_tpu_all",
    define_values = {
        "MEDIAPIPE_EDGE_TPU": "all",
    },
)

cc_library(
    name = "tflite_inference_calculator",
    srcs = ["tflite_inference_calculator.cc"],
    copts = select({
        "//mediapipe:ios": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    defines = select({
        "//conditions:default": [],
        ":edge_tpu_usb": ["MEDIAPIPE_EDGE_TPU=usb"],
        ":edge_tpu_pci": ["MEDIAPIPE_EDGE_TPU=pci"],
        ":edge_tpu_all": ["MEDIAPIPE_EDGE_TPU=all"],
    }),
    deps = [
        ":tflite_inference_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/stream_handler:fixed_size_input_stream_handler",
        "//mediapipe/util/tflite:config",
        "//mediapipe/util/tflite:tflite_model_loader",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ] + selects.with_or({
        ":gpu_inference_disabled": [],
        "//mediapipe:ios": [
            "//mediapipe/gpu:MPPMetalHelper",
            "//mediapipe/gpu:MPPMetalUtil",
            "//mediapipe/gpu:gpu_buffer",
            "//mediapipe/objc:mediapipe_framework_ios",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/common:shape",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/metal:buffer_convert",
            "@org_tensorflow//tensorflow/lite/delegates/gpu:metal_delegate",
            "@org_tensorflow//tensorflow/lite/delegates/gpu:metal_delegate_internal",
            "//third_party/apple_frameworks:MetalKit",
            "//third_party/apple_frameworks:CoreVideo",
        ],
        "//conditions:default": [
            "//mediapipe/util/tflite:tflite_gpu_runner",
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gpu_buffer",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/common:shape",
            "@org_tensorflow//tensorflow/lite/delegates/gpu:gl_delegate",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_buffer",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_program",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_shader",
        ],
    }) + select({
        "//conditions:default": [],
        "//mediapipe:android": [
            "//mediapipe/util/android/file/base",
            "@org_tensorflow//tensorflow/lite/delegates/nnapi:nnapi_delegate",
        ],
    }) + select({
        "//mediapipe:emscripten": [],
        "//conditions:default": [
            "//mediapipe/util:cpu_util",
        ],
    }) + select({
        "//conditions:default": [],
        ":edge_tpu_usb": [
            "@libedgetpu//tflite/public:edgetpu",
            "@libedgetpu//tflite/public:oss_edgetpu_direct_usb",
        ],
        ":edge_tpu_pci": [
            "@libedgetpu//tflite/public:edgetpu",
            "@libedgetpu//tflite/public:oss_edgetpu_direct_pci",
        ],
        ":edge_tpu_all": [
            "@libedgetpu//tflite/public:edgetpu",
            "@libedgetpu//tflite/public:oss_edgetpu_direct_all",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "tflite_converter_calculator",
    srcs = ["tflite_converter_calculator.cc"],
    copts = select({
        "//mediapipe:ios": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":tflite_converter_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/stream_handler:fixed_size_input_stream_handler",
        "//mediapipe/util:resource_util",
        "//mediapipe/util/tflite:config",
        "@com_google_absl//absl/log:absl_check",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ] + selects.with_or({
        ":gpu_inference_disabled": [],
        "//mediapipe:ios": [
            "//mediapipe/gpu:MPPMetalUtil",
            "//mediapipe/gpu:MPPMetalHelper",
            "//mediapipe/objc:mediapipe_framework_ios",
            "@org_tensorflow//tensorflow/lite/delegates/gpu:metal_delegate",
            "//third_party/apple_frameworks:MetalKit",
            "//third_party/apple_frameworks:CoreVideo",
        ],
        "//conditions:default": [
            "//mediapipe/gpu:gl_calculator_helper",
            "@org_tensorflow//tensorflow/lite/delegates/gpu:gl_delegate",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_buffer",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_program",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_shader",
        ],
    }) + select({
        "//mediapipe/gpu:disable_gpu": [],
        "//conditions:default": [
            "//mediapipe/gpu:gpu_buffer",
        ],
    }),
    alwayslink = 1,
)

# TODO: Re-evaluate which of these libraries we can avoid making
# cc_library_with_tflite and can be changed back to cc_library.
cc_library_with_tflite(
    name = "tflite_model_calculator",
    srcs = ["tflite_model_calculator.cc"],
    tflite_deps = [
        "@org_tensorflow//tensorflow/lite:framework_stable",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:resources",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tflite_tensors_to_segmentation_calculator",
    srcs = ["tflite_tensors_to_segmentation_calculator.cc"],
    deps = [
        ":tflite_tensors_to_segmentation_calculator_cc_proto",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/util:resource_util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/lite:framework",
    ] + selects.with_or({
        ":gpu_inference_disabled": [],
        "//mediapipe:ios": [],
        "//conditions:default": [
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gl_simple_shaders",
            "//mediapipe/gpu:gpu_buffer",
            "//mediapipe/gpu:shader_util",
            "@org_tensorflow//tensorflow/lite/delegates/gpu:gl_delegate",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_buffer",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_program",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_shader",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_texture",
        ],
    }),
    alwayslink = 1,
)

cc_test(
    name = "tflite_tensors_to_classification_calculator_test",
    srcs = ["tflite_tensors_to_classification_calculator_test.cc"],
    data = ["testdata/labelmap.txt"],
    deps = [
        ":tflite_tensors_to_classification_calculator",
        ":tflite_tensors_to_classification_calculator_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

cc_library(
    name = "tflite_tensors_to_detections_calculator",
    srcs = ["tflite_tensors_to_detections_calculator.cc"],
    copts = select({
        "//mediapipe:ios": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":tflite_tensors_to_detections_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location",
        "//mediapipe/framework/formats/object_detection:anchor_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/util/tflite:config",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/lite:framework",
    ] + selects.with_or({
        ":gpu_inference_disabled": [],
        "//mediapipe:ios": [
            "//mediapipe/gpu:MPPMetalUtil",
            "//mediapipe/gpu:gpu_buffer",
            "//mediapipe/gpu:MPPMetalHelper",
            "//mediapipe/objc:mediapipe_framework_ios",
            "@org_tensorflow//tensorflow/lite/delegates/gpu:metal_delegate",
            "//third_party/apple_frameworks:MetalKit",
            "//third_party/apple_frameworks:CoreVideo",
        ],
        "//conditions:default": [
            "//mediapipe/gpu:gl_calculator_helper",
            "@org_tensorflow//tensorflow/lite/delegates/gpu:gl_delegate",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_buffer",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_program",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_shader",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "tflite_tensors_to_classification_calculator",
    srcs = ["tflite_tensors_to_classification_calculator.cc"],
    deps = [
        ":tflite_tensors_to_classification_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:location",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/util:resource_util",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/lite:framework",
    ] + select({
        "//mediapipe:android": [
            "//mediapipe/util/android/file/base",
        ],
        "//mediapipe:ios": [
            "//mediapipe/util/android/file/base",
        ],
        "//mediapipe:emscripten": [
            "//mediapipe/util/android/file/base",
        ],
        "//mediapipe:macos": [
            "//mediapipe/framework/port:file_helpers",
        ],
        "//conditions:default": [
            "//mediapipe/framework/port:file_helpers",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "tflite_tensors_to_landmarks_calculator",
    srcs = ["tflite_tensors_to_landmarks_calculator.cc"],
    deps = [
        ":tflite_tensors_to_landmarks_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/log:absl_check",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tflite_tensors_to_floats_calculator",
    srcs = ["tflite_tensors_to_floats_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
    alwayslink = 1,
)

# To run this with native GPU on Linux, use:
# bazel test //mediapipe/calculators/tflite:tflite_inference_calculator_test --copt=-DTFLITE_GPU_EXTRA_GLES_DEPS --copt=-DMESA_EGL_NO_X11_HEADERS --copt=-DEGL_NO_X11 --config=grte_v5 --test_strategy=local
cc_test(
    name = "tflite_inference_calculator_test",
    srcs = [
        "tflite_inference_calculator_test.cc",
        "tflite_inference_calculator_test_common.h",
    ],
    data = ["testdata/add.bin"],
    linkstatic = 1,
    deps = [
        ":tflite_inference_calculator",
        ":tflite_inference_calculator_cc_proto",
        ":tflite_model_calculator",
        "//mediapipe/calculators/util:resource_provider_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/tool:validate_type",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:type_to_tflitetype",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
    ],
)

cc_test(
    name = "tflite_converter_calculator_test",
    srcs = ["tflite_converter_calculator_test.cc"],
    deps = [
        ":tflite_converter_calculator",
        ":tflite_converter_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/tool:validate_type",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

cc_test(
    name = "tflite_model_calculator_test",
    srcs = ["tflite_model_calculator_test.cc"],
    data = ["testdata/add.bin"],
    deps = [
        ":tflite_model_calculator",
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/util:local_file_contents_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:resources",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "tflite_custom_op_resolver_calculator_test",
    srcs = ["tflite_custom_op_resolver_calculator_test.cc"],
    data = ["testdata/add.bin"],
    deps = [
        ":tflite_custom_op_resolver_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)
