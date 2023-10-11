# Copyright 2022 The MediaPipe Authors.
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

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

# Docs for Mediapipe Tasks Text Embedder
# https://developers.google.com/mediapipe/solutions/text/text_embedder
cc_library(
    name = "text_embedder",
    srcs = ["text_embedder.cc"],
    hdrs = ["text_embedder.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":text_embedder_graph",
        "//mediapipe/calculators/tensor:inference_calculator_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/tasks/cc/components/containers:embedding_result",
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_cc_proto",
        "//mediapipe/tasks/cc/components/processors:embedder_options",
        "//mediapipe/tasks/cc/components/processors/proto:embedder_options_cc_proto",
        "//mediapipe/tasks/cc/components/utils:cosine_similarity",
        "//mediapipe/tasks/cc/core:base_options",
        "//mediapipe/tasks/cc/core:base_task_api",
        "//mediapipe/tasks/cc/core:task_api_factory",
        "//mediapipe/tasks/cc/core/proto:base_options_cc_proto",
        "//mediapipe/tasks/cc/text/text_embedder/proto:text_embedder_graph_options_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "text_embedder_graph",
    srcs = ["text_embedder_graph.cc"],
    deps = [
        "//mediapipe/calculators/tensor:inference_calculator_cc_proto",
        "//mediapipe/calculators/tensor:inference_calculator_cpu",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:port",
        "//mediapipe/tasks/cc/components/calculators:tensors_to_embeddings_calculator_cc_proto",
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_cc_proto",
        "//mediapipe/tasks/cc/components/processors:embedding_postprocessing_graph",
        "//mediapipe/tasks/cc/components/processors:text_preprocessing_graph",
        "//mediapipe/tasks/cc/components/processors/proto:embedding_postprocessing_graph_options_cc_proto",
        "//mediapipe/tasks/cc/components/processors/proto:text_model_type_cc_proto",
        "//mediapipe/tasks/cc/components/processors/proto:text_preprocessing_graph_options_cc_proto",
        "//mediapipe/tasks/cc/core:model_resources",
        "//mediapipe/tasks/cc/core:model_task_graph",
        "//mediapipe/tasks/cc/core/proto:model_resources_calculator_cc_proto",
        "//mediapipe/tasks/cc/text/text_embedder/proto:text_embedder_graph_options_cc_proto",
        "//mediapipe/tasks/cc/text/utils:text_model_utils",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_test(
    name = "text_embedder_test",
    srcs = ["text_embedder_test.cc"],
    data = [
        "//mediapipe/tasks/testdata/text:mobilebert_embedding_model",
        "//mediapipe/tasks/testdata/text:regex_embedding_with_metadata",
        "//mediapipe/tasks/testdata/text:universal_sentence_encoder_qa",
    ],
    tags = ["not_run:arm"],
    deps = [
        ":text_embedder",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/containers:embedding_result",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_sentencepiece//:sentencepiece_processor",
        "@org_tensorflow//tensorflow/lite:test_util",
    ],
)
