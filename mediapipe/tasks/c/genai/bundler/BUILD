# Copyright 2026 The MediaPipe Authors.
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

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

config_setting(
    name = "enable_odml_converter",
    define_values = {
        "ENABLE_ODML_CONVERTER": "1",
    },
)

cc_library(
    name = "llm_bundler_utils_c_lib",
    srcs = ["llm_bundler_utils.cc"],
    hdrs = ["llm_bundler_utils.h"],
    copts = select({
        ":enable_odml_converter": ["-DENABLE_ODML_CONVERTER"],
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_absl//absl/log:absl_log",
    ] + select({
        ":enable_odml_converter": [
            "@odml//odml/infra/genai/inference/proto:llm_params_cc_proto",
            "@odml//odml/infra/genai/inference/proto:prompt_template_cc_proto",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)
