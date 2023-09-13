# Copyright 2022 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Placeholder for internal Python strict library and test compatibility macro.
# Placeholder for internal Python strict test compatibility macro.

licenses(["notice"])

package(default_visibility = ["//mediapipe:__subpackages__"])

filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/**",
    ]),
)

py_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.py"],
    deps = [
        ":model_util",
        "//mediapipe/model_maker/python/core/data:dataset",
    ],
)

py_library(
    name = "model_util",
    srcs = ["model_util.py"],
    deps = [
        ":quantization",
        "//mediapipe/model_maker/python/core/data:dataset",
    ],
)

py_test(
    name = "model_util_test",
    srcs = ["model_util_test.py"],
    deps = [
        ":model_util",
        ":quantization",
        ":test_util",
    ],
)

py_library(
    name = "file_util",
    srcs = ["file_util.py"],
)

py_test(
    name = "file_util_test",
    srcs = ["file_util_test.py"],
    data = [":testdata"],
    tags = ["requires-net:external"],
    deps = [":file_util"],
)

py_library(
    name = "hub_loader",
    srcs = ["hub_loader.py"],
)

py_test(
    name = "hub_loader_test",
    srcs = ["hub_loader_test.py"],
    data = [":testdata"],
    deps = [
        ":hub_loader",
        "//mediapipe/tasks/python/test:test_utils",
    ],
)

py_library(
    name = "loss_functions",
    srcs = ["loss_functions.py"],
    srcs_version = "PY3",
    deps = [
        ":file_util",
        ":model_util",
    ],
)

py_test(
    name = "loss_functions_test",
    srcs = ["loss_functions_test.py"],
    tags = [
        "requires-net:external",
    ],
    deps = [":loss_functions"],
)

######################################################################
# Public target of the MediaPipe Model Maker Quantization Config.

# Quantization Config is used to export a quantized model. Please refer
# to the specific task documentations such as:
# https://developers.google.com/mediapipe/solutions/vision/image_classifier/customize
# for usage information.
######################################################################
py_library(
    name = "metrics",
    srcs = ["metrics.py"],
)

py_test(
    name = "metrics_test",
    srcs = ["metrics_test.py"],
    deps = [":metrics"],
)

py_library(
    name = "quantization",
    srcs = ["quantization.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/model_maker/python/core/data:dataset"],
)

py_test(
    name = "quantization_test",
    srcs = ["quantization_test.py"],
    deps = [
        ":quantization",
        ":test_util",
    ],
)
