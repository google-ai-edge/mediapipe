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

cc_library(
    name = "llm_bundler_utils",
    srcs = ["llm_bundler_utils.cc"],
    hdrs = ["llm_bundler_utils.h"],
    deps = [
        "//mediapipe/tasks/cc/genai/inference/proto:llm_params_cc_proto",
        "//mediapipe/tasks/cc/genai/inference/proto:prompt_template_cc_proto",
    ],
)

cc_library(
    name = "llm_bundler_utils_c_lib",
    srcs = ["llm_bundler_utils.cc"],
    hdrs = ["llm_bundler_utils.h"],
    deps = [
        "//mediapipe/tasks/cc/genai/inference/proto:llm_params_cc_proto",
        "//mediapipe/tasks/cc/genai/inference/proto:prompt_template_cc_proto",
    ],
    alwayslink = 1,
)
