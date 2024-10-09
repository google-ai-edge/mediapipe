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

py_library(
    name = "data_util",
    srcs = ["data_util.py"],
    deps = [
        "@model_maker_pip_deps_numpy//:pkg",
        "@model_maker_pip_deps_opencv_python//:pkg",
        "@model_maker_pip_deps_tensorflow//:pkg",
    ],
)

py_test(
    name = "data_util_test",
    srcs = ["data_util_test.py"],
    data = ["//mediapipe/model_maker/python/core/data/testdata"],
    deps = [
        ":data_util",
        "@model_maker_pip_deps_absl_py//:pkg",
        "@model_maker_pip_deps_tensorflow//:pkg",
    ],
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    srcs_version = "PY3",
    deps = ["@model_maker_pip_deps_tensorflow//:pkg"],
)

py_test(
    name = "dataset_test",
    srcs = ["dataset_test.py"],
    deps = [
        ":dataset",
        "//mediapipe/model_maker/python/core/utils:test_util",
        "@model_maker_pip_deps_numpy//:pkg",
        "@model_maker_pip_deps_tensorflow//:pkg",
    ],
)

py_library(
    name = "classification_dataset",
    srcs = ["classification_dataset.py"],
    deps = [
        ":dataset",
        "@model_maker_pip_deps_tensorflow//:pkg",
    ],
)

py_test(
    name = "classification_dataset_test",
    srcs = ["classification_dataset_test.py"],
    deps = [
        ":classification_dataset",
        "@model_maker_pip_deps_tensorflow//:pkg",
    ],
)

py_library(
    name = "cache_files",
    srcs = ["cache_files.py"],
    deps = [
        "@model_maker_pip_deps_pyyaml//:pkg",
        "@model_maker_pip_deps_tensorflow//:pkg",
    ],
)

py_test(
    name = "cache_files_test",
    srcs = ["cache_files_test.py"],
    deps = [
        ":cache_files",
        "@model_maker_pip_deps_tensorflow//:pkg",
    ],
)
