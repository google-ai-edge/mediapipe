# Copyright (c) 2023 Intel Corporation
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

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:private"])

mediapipe_proto_library(
    name = "openvino_inference_calculator_proto",
    srcs = ["openvino_inference_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "openvino_converter_calculator_proto",
    srcs = ["openvino_converter_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "openvino_tensors_to_classification_calculator_proto",
    srcs = ["openvino_tensors_to_classification_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "openvino_tensors_to_detections_calculator_proto",
    srcs = ["openvino_tensors_to_detections_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "openvino_converter_calculator",
    srcs = ["openvino_converter_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":openvino_converter_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/stream_handler:fixed_size_input_stream_handler",
        "//mediapipe/util:resource_util",
        "//third_party:openvino",
    ],
    alwayslink = 1,
)

cc_library(
    name = "openvino_internal_utils",
    srcs = ["internal/infer_request_wrap.hpp"],
    #    srcs = glob([
    #        "internal/*.cpp",
    #        "internal/*.hpp",
    #    ]),
    visibility = ["//visibility:public"],
    deps = [
        "//third_party:openvino",
    ],
)

cc_library(
    name = "openvino_inference_calculator",
    srcs = ["openvino_inference_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":openvino_internal_utils",
        ":openvino_inference_calculator_cc_proto",
        "//mediapipe/framework:timestamp",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/tool:status_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "//mediapipe/framework/deps:clock",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:map_util",
        "//mediapipe/framework:packet",
        "//third_party:openvino",
    ] + select({
        "//conditions:default": [
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "openvino_tensors_to_classification_calculator",
    srcs = ["openvino_tensors_to_classification_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":openvino_tensors_to_classification_calculator_cc_proto",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:location",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/util:resource_util",
        "//third_party:openvino",
    ] + select({
        "//mediapipe:android": [
            "//mediapipe/util/android/file/base",
        ],
        "//mediapipe:ios": [
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
    name = "openvino_tensors_to_detections_calculator",
    srcs = ["openvino_tensors_to_detections_calculator.cc"],
    copts = select({
        "//mediapipe:ios": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    linkopts = select({
        "//mediapipe:ios": [
            "-framework CoreVideo",
            "-framework MetalKit",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":openvino_tensors_to_detections_calculator_cc_proto",
        "//mediapipe/framework/formats:detection_cc_proto",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:location",
        "//mediapipe/framework/formats/object_detection:anchor_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//third_party:openvino",
    ],
    alwayslink = 1,
)

# To run this with native GPU on Linux, use:
# bazel test //mediapipe/calculators/tflite:tflite_inference_calculator_test --copt=-DTFLITE_GPU_EXTRA_GLES_DEPS --copt=-DMESA_EGL_NO_X11_HEADERS --copt=-DEGL_NO_X11 --config=grte_v5 --test_strategy=local
cc_test(
    name = "openvino_inference_calculator_test",
    srcs = [
        "openvino_inference_calculator_test.cc",
        "openvino_inference_calculator_test_common.h",
    ],
    data = [
        "testdata/add.bin",
        "testdata/add.xml",
    ],
    linkstatic = 1,
    deps = [
        ":openvino_inference_calculator",
        ":openvino_inference_calculator_cc_proto",
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/util:local_file_contents_calculator",
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
    ],
)

cc_test(
    name = "openvino_converter_calculator_test",
    srcs = ["openvino_converter_calculator_test.cc"],
    deps = [
        ":openvino_converter_calculator",
        ":openvino_converter_calculator_cc_proto",
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
    ],
)

cc_test(
    name = "openvino_tensors_to_classification_calculator_test",
    srcs = ["openvino_tensors_to_classification_calculator_test.cc"],
    data = ["testdata/labelmap.txt"],
    deps = [
        ":openvino_tensors_to_classification_calculator",
        ":openvino_tensors_to_classification_calculator_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)
