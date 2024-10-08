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

# Utilities to help optimize Large Language Models (LLM).

package(default_visibility = [
    "//mediapipe/tasks/cc:__subpackages__",
])

cc_library(
    name = "memory_mapped_file",
    srcs = select({
        "//mediapipe:windows": ["memory_mapped_file_win.cc"],
        "//conditions:default": ["memory_mapped_file_posix.cc"],
    }),
    hdrs = ["memory_mapped_file.h"],
    copts = [
        "-D_WIN32_WINNT=_WIN32_WINNT_WIN10",
    ],
    deps = [
        ":scoped_file",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "scoped_file",
    srcs = select({
        "//mediapipe:windows": ["scoped_file_win.cc"],
        "//conditions:default": ["scoped_file_posix.cc"],
    }),
    hdrs = ["scoped_file.h"],
    deps = [
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "metadata_utils",
    srcs = ["metadata_utils.cc"],
    hdrs = ["metadata_utils.h"],
    deps = [
        "//mediapipe/tasks/cc/genai/inference/proto:llm_params_cc_proto",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@org_tensorflow//tensorflow/lite:framework_stable",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "model_data",
    srcs = ["model_data.cc"],
    hdrs = ["model_data.h"],
    deps = [
        ":memory_mapped_file",
        ":metadata_utils",
        ":scoped_file",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc/genai/inference/proto:llm_file_metadata_cc_proto",
        "//mediapipe/tasks/cc/genai/inference/proto:llm_params_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@flatbuffers//:runtime_cc",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "well_known_models",
    srcs = ["well_known_models.cc"],
    hdrs = ["well_known_models.h"],
    deps = [
        "//mediapipe/tasks/cc/genai/inference/proto:llm_params_cc_proto",
        "//mediapipe/tasks/cc/genai/inference/proto:transformer_params_cc_proto",
    ],
)
