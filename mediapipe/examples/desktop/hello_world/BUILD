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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "hello_world",
    srcs = ["hello_world.cc"],
    deps = [
        "//mediapipe/calculators/core:pass_through_calculator",
        "//mediapipe/framework:calculator_graph",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
    ],
)
