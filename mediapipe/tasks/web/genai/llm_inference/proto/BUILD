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

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

package(
    default_visibility = [
        "//mediapipe/tasks:internal",
        "//mediapipe/web/graph_runner:__subpackages__",
    ],
)

licenses(["notice"])

mediapipe_proto_library(
    name = "llm_inference_graph_options_proto",
    srcs = ["llm_inference_graph_options.proto"],
    deps = [
        ":sampler_params_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_proto",
    ],
)

mediapipe_proto_library(
    name = "transformer_params_proto",
    srcs = ["transformer_params.proto"],
)

mediapipe_proto_library(
    name = "llm_params_proto",
    srcs = ["llm_params.proto"],
    deps = [
        ":prompt_template_proto",
        ":transformer_params_proto",
    ],
)

mediapipe_proto_library(
    name = "sampler_params_proto",
    srcs = ["sampler_params.proto"],
)

mediapipe_proto_library(
    name = "prompt_template_proto",
    srcs = ["prompt_template.proto"],
)

mediapipe_proto_library(
    name = "detokenizer_calculator_proto",
    srcs = ["detokenizer_calculator.proto"],
)

mediapipe_proto_library(
    name = "llm_gpu_calculator_proto",
    srcs = ["llm_gpu_calculator.proto"],
    deps = [
        ":llm_params_proto",
        ":sampler_params_proto",
    ],
)

mediapipe_proto_library(
    name = "tokenizer_calculator_proto",
    srcs = ["tokenizer_calculator.proto"],
)
