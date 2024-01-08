#
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

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "ovms_calculator",
    srcs = [
    ],
    deps = [
        "//mediapipe/calculators/openvino:openvino_tensors_to_classification_calculator_cc_proto",
        "//mediapipe/calculators/openvino:openvino_tensors_to_detections_calculator_cc_proto",
        "//mediapipe/calculators/openvino:openvino_converter_calculator_cc_proto",
        "//mediapipe/calculators/openvino:openvino_converter_calculator",
        "//mediapipe/calculators/openvino:openvino_tensors_to_classification_calculator",
        "//mediapipe/calculators/openvino:openvino_tensors_to_detections_calculator",
        ":modelapiovmsadapter",
        ":openvinoinferencecalculator",
        ":openvinomodelserversessioncalculator",
    ],
    copts = ["-Iexternal/ovms/src","-Isrc"],
    linkopts = ["-Lmediapipe/"],
    alwayslink = 1,
)

cc_library(
    name = "modelapiovmsadapter",
    srcs = [
        "modelapiovmsadapter.cc"
    ],
    hdrs = [
        "modelapiovmsadapter.hpp"
    ],
    deps = [
        "//mediapipe/framework/port:logging",
        "@ovms//src:ovms_header",
        "@model_api//:model_api",
        "@linux_openvino//:openvino",
    ],
    copts = ["-Iexternal/ovms/src","-Isrc"],
)

cc_library(
    name = "openvinoinferencecalculator",
    srcs = [
        "openvinoinferencecalculator.cc"
    ],
    hdrs = [
        "openvinoinferencecalculator.h"
    ],
    deps = [
        ":modelapiovmsadapter",
        ":openvinoinferencecalculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:tensor", # Tensor GetContract
        "@linux_openvino//:openvino",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/lite/c:c_api",
    ],
    copts = ["-Iexternal/ovms/src","-Isrc"],
    alwayslink = 1,
)

cc_library(
    name = "openvinomodelserversessioncalculator",
    srcs = [
        "openvinomodelserversessioncalculator.cc"
    ],
    hdrs = [
        "openvinomodelserversessioncalculator.h"
    ],
    deps = [
        ":modelapiovmsadapter",
        ":openvinomodelserversessioncalculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "@linux_openvino//:openvino",
    ],
    copts = ["-Iexternal/ovms/src","-Isrc"],
    alwayslink = 1,
)

load("@mediapipe//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library", "mediapipe_proto_library")

mediapipe_proto_library(
    name = "openvinoinferencecalculator_proto",
    srcs = ["openvinoinferencecalculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)
mediapipe_proto_library(
    name = "openvinomodelserversessioncalculator_proto",
    srcs = ["openvinomodelserversessioncalculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)

cc_test(
    name = "openvinoinferencecalculator_test",
    srcs = ["openvinoinferencecalculator_test.cc",
    ],
    deps = [
        ":ovms_calculator",
        "@ovms//src:ovms_header",
        "@ovms//src:ovms_lib",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/stream_handler:sync_set_input_stream_handler"
    ],
    copts = ["-Iexternal/ovms/src","-Isrc"],
)

cc_test(
    name = "openvinomodelserversessioncalculator_test",
    srcs = ["openvinomodelserversessioncalculator_test.cc",
    ],
    deps = [
        ":ovms_calculator",
        "@ovms//src:ovms_header",
        "@ovms//src:ovms_lib",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
    ],
    data = [
        "test_data/config.json",
        "test_data/dummy/1/dummy.bin",
        "test_data/dummy/1/dummy.xml",
        "test_data/add_two_inputs_model/1/add.bin",
        "test_data/add_two_inputs_model/1/add.xml",
    ],
    copts = ["-Iexternal/ovms/src","-Isrc"],
)
