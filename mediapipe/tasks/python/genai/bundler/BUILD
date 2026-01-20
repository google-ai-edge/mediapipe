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

# Placeholder for internal Python strict library and test compatibility macro.
# Placeholder for internal Python strict test compatibility macro.

package(default_visibility = [
    "//third_party/py/ai_edge_torch/google/model_conversion:__subpackages__",
    "//third_party/py/litert_torch/google/model_conversion:__subpackages__",
])

py_library(
    name = "llm_bundler_metadata_options_c",
    srcs = ["llm_bundler_metadata_options_c.py"],
)

py_library(
    name = "llm_bundler_metadata_options",
    srcs = ["llm_bundler_metadata_options.py"],
    deps = [":llm_bundler_metadata_options_c"],
)

py_library(
    name = "llm_bundler",
    srcs = ["llm_bundler.py"],
    deps = [
        ":llm_bundler_metadata_options",
        ":llm_bundler_metadata_options_c",
        "//mediapipe/tasks/python/core:mediapipe_c_bindings",
        "//mediapipe/tasks/python/core:mediapipe_c_utils",
        "//mediapipe/tasks/python/core:serial_dispatcher",
        "//mediapipe/tasks/python/genai/converter:external_dependencies",
        "//mediapipe/tasks/python/metadata/metadata_writers:model_asset_bundle_utils",
    ],
)

py_test(
    name = "llm_bundler_test",
    srcs = ["llm_bundler_test.py"],
    deps = [
        ":llm_bundler",
        "//mediapipe/tasks/cc/genai/inference/proto:llm_params_py_pb2",
        "@com_google_sentencepiece//:sentencepiece_model_py_pb2",
        "@mediapipe_pip_deps_absl_py//:pkg",
    ],
)

py_test(
    name = "llm_bundler_metadata_options_test",
    srcs = ["llm_bundler_metadata_options_test.py"],
    deps = [
        ":llm_bundler_metadata_options",
        "@mediapipe_pip_deps_absl_py//:pkg",
    ],
)
