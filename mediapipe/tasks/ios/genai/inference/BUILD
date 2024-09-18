# Copyright 2024 The MediaPipe Authors. All Rights Reserved.
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

load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

licenses(["notice"])

package(default_visibility = ["//mediapipe/tasks:internal"])

swift_library(
    name = "LlmInference",
    srcs = [
        "sources/LlmInference.swift",
        "sources/LlmInference+Session.swift",
        "//mediapipe/tasks/ios/genai/core:sources/GenAiInferenceError.swift",
        "//mediapipe/tasks/ios/genai/core:sources/LlmSessionRunner.swift",
        "//mediapipe/tasks/ios/genai/core:sources/LlmTaskRunner.swift",
    ],
    # This ensures the compiler does not complain about MediaPipeTasksGenAIC being built separately.
    copts = ["-no-verify-emitted-module-interface"],
    generated_header_name = "LlmInference-Swift.h",
    generates_header = 1,
    module_name = "MediaPipeTasksGenAI",
    deps = [
        "//mediapipe/tasks/cc/genai/inference/c:libllm_inference_engine_cpu",
        "//third_party/apple_frameworks:Foundation",
    ],
)
