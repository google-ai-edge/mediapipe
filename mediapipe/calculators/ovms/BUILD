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
    srcs = ["modelapiovmsadapter.cc",
            "modelapiovmsadapter.hpp",
            "modelapiovmsinferencecalculator.cc",
            "modelapiovmssessioncalculator.cc"],
    deps = [
        "//mediapipe/calculators/ovms:modelapiovmsinferencecalculator_cc_proto",
        "//mediapipe/calculators/ovms:modelapiovmssessioncalculator_cc_proto",
        "//mediapipe/calculators/openvino:openvino_tensors_to_classification_calculator_cc_proto",
        "//mediapipe/calculators/openvino:openvino_tensors_to_detections_calculator_cc_proto",
        "//mediapipe/calculators/openvino:openvino_converter_calculator_cc_proto",
        "//mediapipe/calculators/openvino:openvino_converter_calculator",
        "//mediapipe/calculators/openvino:openvino_tensors_to_classification_calculator",
        "//mediapipe/calculators/openvino:openvino_tensors_to_detections_calculator",
        "//mediapipe/framework:calculator_framework",
        "@org_tensorflow//tensorflow/core:framework",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/formats:tensor", # Tensor GetContract
        "@ovms//src:ovms_header",
        "@model_api//:adapter_api",
        "@org_tensorflow//tensorflow/lite:framework_stable", # to use tflite
    ],
    copts = ["-Iexternal/ovms/src","-Isrc"],
    linkopts = ["-Lmediapipe/"],
    alwayslink = 1,
)

load("@mediapipe//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library", "mediapipe_proto_library")

# ovms
mediapipe_proto_library(
    name = "modelapiovmsinferencecalculator_proto",
    srcs = ["modelapiovmsinferencecalculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)
mediapipe_proto_library(
    name = "modelapiovmssessioncalculator_proto",
    srcs = ["modelapiovmssessioncalculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)
