# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
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

# Placeholder for internal Python strict test compatibility macro.
# Placeholder for internal Python strict library and test compatibility macro.

licenses(["notice"])

package(default_visibility = ["//mediapipe:__subpackages__"])

filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/**",
    ]),
)

py_library(
    name = "hyperparameters",
    srcs = ["hyperparameters.py"],
    deps = [
        "//mediapipe/model_maker/python/core:hyperparameters",
    ],
)

py_library(
    name = "model_options",
    srcs = ["model_options.py"],
)

py_library(
    name = "model_spec",
    srcs = ["model_spec.py"],
)

py_test(
    name = "model_spec_test",
    srcs = ["model_spec_test.py"],
    deps = [":model_spec"],
)

py_library(
    name = "face_stylizer_options",
    srcs = ["face_stylizer_options.py"],
    deps = [
        ":hyperparameters",
        ":model_options",
        ":model_spec",
    ],
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    deps = [
        "//mediapipe/model_maker/python/core/data:classification_dataset",
        "//mediapipe/model_maker/python/vision/core:image_utils",
    ],
)

py_test(
    name = "dataset_test",
    srcs = ["dataset_test.py"],
    data = [
        ":testdata",
    ],
    deps = [
        ":dataset",
        "//mediapipe/tasks/python/test:test_utils",
    ],
)
