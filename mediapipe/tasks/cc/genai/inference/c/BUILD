# Copyright 2024 The MediaPipe Authors.
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

package(default_visibility = [
    "//mediapipe/tasks/ios:__subpackages__",
    "//mediapipe/tasks/java/com/google/mediapipe/tasks:__subpackages__",
])

cc_library(
    name = "llm_inference_engine_hdr",
    hdrs = ["llm_inference_engine.h"],
)

cc_library(
    name = "libllm_inference_engine_cpu",
    srcs = ["llm_inference_engine_cpu.cc"],
    hdrs = ["llm_inference_engine.h"],
    tags = ["swift_module=MediaPipeTasksGenAIC"],
    deps = [
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc/genai/inference/proto:llm_params_cc_proto",
        "//mediapipe/tasks/cc/genai/inference/proto:transformer_params_cc_proto",
        "//mediapipe/tasks/cc/genai/inference/utils/llm_utils:memory_mapped_file",
        "//mediapipe/tasks/cc/genai/inference/utils/llm_utils:metadata_utils",
        "//mediapipe/tasks/cc/genai/inference/utils/llm_utils:model_data",
        "//mediapipe/tasks/cc/genai/inference/utils/llm_utils:scoped_file",
        "//mediapipe/tasks/cc/genai/inference/utils/xnn_utils:graph_builder",
        "//mediapipe/tasks/cc/genai/inference/utils/xnn_utils:llm",
        "//mediapipe/tasks/cc/genai/inference/utils/xnn_utils:llm_builder_factory",
        "//mediapipe/tasks/cc/genai/inference/utils/xnn_utils:llm_weights",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_sentencepiece//:sentencepiece_processor",
        "@org_tensorflow//tensorflow/lite:framework_stable",
    ],
)

cc_binary(
    name = "llm_inference_engine_cpu_main",
    srcs = ["llm_inference_engine_cpu_main.cc"],
    deps = [
        ":libllm_inference_engine_cpu",
        "//mediapipe/framework/deps:file_path",
        "//third_party:glog",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
    ],
)
