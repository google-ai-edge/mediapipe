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
    "//mediapipe:__subpackages__",
    "//third_party/odml:__subpackages__",
])

py_library(
    name = "converter_base",
    srcs = ["converter_base.py"],
)

py_library(
    name = "converter_factory",
    srcs = ["converter_factory.py"],
    deps = [
        ":converter_base",
        ":pytorch_converter",
        ":safetensors_converter",
        ":weight_bins_writer",
    ],
)

py_library(
    name = "llm_converter",
    srcs = ["llm_converter.py"],
    deps = [
        ":converter_base",
        ":converter_factory",
        ":quantization_util",
        "//mediapipe/python:_framework_bindings",
    ],
)

py_library(
    name = "pytorch_converter",
    srcs = ["pytorch_converter.py"],
    deps = [
        ":converter_base",
    ],
)

py_library(
    name = "safetensors_converter",
    srcs = ["safetensors_converter.py"],
    deps = [
        ":converter_base",
    ],
)

py_library(
    name = "quantization_util",
    srcs = ["quantization_util.py"],
)

py_test(
    name = "quantization_util_test",
    srcs = ["quantization_util_test.py"],
    srcs_version = "PY3",
    deps = [
        ":quantization_util",
    ],
)

py_library(
    name = "weight_bins_writer",
    srcs = ["weight_bins_writer.py"],
    deps = [
        ":converter_base",
        ":quantization_util",
    ],
)

py_test(
    name = "weight_bins_writer_test",
    srcs = ["weight_bins_writer_test.py"],
    srcs_version = "PY3",
    deps = [
        ":weight_bins_writer",
    ],
)
