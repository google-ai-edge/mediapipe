# Copyright 2023 The MediaPipe Authors.
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
# Placeholder for internal Python GPU test rule.

licenses(["notice"])

package(default_visibility = ["//mediapipe:__subpackages__"])

py_library(
    name = "constants",
    srcs = ["constants.py"],
    deps = ["//mediapipe/model_maker/python/core/utils:file_util"],
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
    deps = ["//mediapipe/model_maker/python/core/utils:loss_functions"],
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
        ":constants",
        "//mediapipe/model_maker/python/core/data:classification_dataset",
        "//mediapipe/python:_framework_bindings",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/vision:face_aligner",
    ],
)

py_library(
    name = "face_stylizer",
    srcs = ["face_stylizer.py"],
    deps = [
        ":constants",
        ":face_stylizer_options",
        ":hyperparameters",
        ":model_options",
        ":model_spec",
        "//mediapipe/model_maker/python/core/data:classification_dataset",
        "//mediapipe/model_maker/python/core/utils:loss_functions",
        "//mediapipe/model_maker/python/core/utils:model_util",
        "//mediapipe/model_maker/python/vision/core:image_preprocessing",
        "//mediapipe/tasks/python/metadata/metadata_writers:face_stylizer",
    ],
)

py_library(
    name = "face_stylizer_import",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":dataset",
        ":face_stylizer",
        ":face_stylizer_options",
        ":hyperparameters",
        ":model_options",
        ":model_spec",
    ],
)
