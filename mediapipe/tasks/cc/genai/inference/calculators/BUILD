# Copyright 2023 The MediaPipe Authors.
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

package(default_visibility = [
    "//mediapipe/tasks:__subpackages__",
    "//third_party/odml/audio/transcription:__subpackages__",
    "//third_party/odml/litert_lm/runtime:__subpackages__",
    "//third_party/odml/vogo:__subpackages__",
])

licenses(["notice"])

mediapipe_proto_library(
    name = "detokenizer_calculator_proto",
    srcs = ["detokenizer_calculator.proto"],
)

mediapipe_proto_library(
    name = "tokenizer_calculator_proto",
    srcs = ["tokenizer_calculator.proto"],
)

mediapipe_proto_library(
    name = "llm_gpu_calculator_proto",
    srcs = ["llm_gpu_calculator.proto"],
    deps = [
        "//mediapipe/tasks/cc/genai/inference/proto:llm_file_metadata_proto",
        "//mediapipe/tasks/cc/genai/inference/proto:llm_params_proto",
        "//mediapipe/tasks/cc/genai/inference/proto:sampler_params_proto",
    ],
)

mediapipe_proto_library(
    name = "model_data_calculator_proto",
    srcs = ["model_data_calculator.proto"],
)
