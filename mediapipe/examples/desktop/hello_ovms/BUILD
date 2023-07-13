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

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "hello_ovms",
    srcs = [
        "hello_ovms.cc",
        "c_api_test_utils.hpp"
    ],
    deps = [
        "//mediapipe/calculators/ovms:ovms_calculator",
        "//mediapipe/framework:calculator_graph",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
    ],
    linkopts = [
        "-lxml2",
        "-luuid",
        "-lstdc++fs",
        "-lcrypto",
    ],
    copts = [
        "-Iexternal/ovms/src",
        "-Iexternal/ovms/src/test",
    ],
)

