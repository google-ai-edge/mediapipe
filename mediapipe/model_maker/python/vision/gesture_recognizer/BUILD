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

# Placeholder for internal Python strict binary and library compatibility macro.
# Placeholder for internal Python strict test compatibility macro.

licenses(["notice"])

package(default_visibility = ["//mediapipe:__subpackages__"])

# TODO: Remove the unnecessary test data once the demo data are moved to an open-sourced
# directory.
filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/**",
    ]),
)

py_library(
    name = "constants",
    srcs = ["constants.py"],
    deps = [
        "//mediapipe/model_maker/python/core/utils:file_util",
    ],
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    deps = [
        ":constants",
        ":metadata_writer",
        "//mediapipe/model_maker/python/core/data:classification_dataset",
        "//mediapipe/model_maker/python/core/utils:model_util",
        "//mediapipe/python:_framework_bindings",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/vision:hand_landmarker",
    ],
)

# TODO: Remove notsan tag once tasks no longer has race condition issue
py_test(
    name = "dataset_test",
    srcs = ["dataset_test.py"],
    data = [":testdata"],
    tags = [
        "not_run:arm",
        "notsan",
        "requires-net:external",
    ],
    deps = [
        ":dataset",
        "//mediapipe/tasks/python/test:test_utils",
        "//mediapipe/tasks/python/vision:hand_landmarker",
    ],
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
    name = "gesture_recognizer_options",
    srcs = ["gesture_recognizer_options.py"],
    deps = [
        ":hyperparameters",
        ":model_options",
    ],
)

py_library(
    name = "gesture_recognizer",
    srcs = ["gesture_recognizer.py"],
    deps = [
        ":constants",
        ":gesture_recognizer_options",
        ":hyperparameters",
        ":metadata_writer",
        ":model_options",
        "//mediapipe/model_maker/python/core/data:classification_dataset",
        "//mediapipe/model_maker/python/core/tasks:classifier",
        "//mediapipe/model_maker/python/core/utils:loss_functions",
        "//mediapipe/model_maker/python/core/utils:model_util",
        "//mediapipe/tasks/python/metadata/metadata_writers:metadata_writer",
    ],
)

######################################################################
# Public target of the MediaPipe Model Maker GestureRecognizer APIs.

# Please see https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer/customize
# for more information about the MediaPipe Model Maker GestureRecognizer APIs.
######################################################################
py_library(
    name = "gesture_recognizer_import",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":dataset",
        ":gesture_recognizer",
        ":gesture_recognizer_options",
        ":hyperparameters",
        ":model_options",
    ],
)

py_library(
    name = "metadata_writer",
    srcs = ["metadata_writer.py"],
    deps = [
        "//mediapipe/tasks/python/metadata/metadata_writers:metadata_writer",
        "//mediapipe/tasks/python/metadata/metadata_writers:model_asset_bundle_utils",
    ],
)

# TODO: Remove notsan tag once tasks no longer has race condition issue
py_test(
    name = "gesture_recognizer_test",
    size = "large",
    srcs = ["gesture_recognizer_test.py"],
    data = [":testdata"],
    shard_count = 2,
    tags = [
        "not_run:arm",
        "notsan",
        "requires-net:external",
    ],
    deps = [
        ":gesture_recognizer_import",
        ":hyperparameters",
        ":model_options",
        "//mediapipe/model_maker/python/core/utils:test_util",
        "//mediapipe/tasks/python/test:test_utils",
    ],
)

py_test(
    name = "metadata_writer_test",
    srcs = ["metadata_writer_test.py"],
    data = [":testdata"],
    deps = [
        ":metadata_writer",
        "//mediapipe/tasks/python/metadata/metadata_writers:metadata_writer",
        "//mediapipe/tasks/python/test:test_utils",
    ],
)

py_binary(
    name = "gesture_recognizer_demo",
    srcs = ["gesture_recognizer_demo.py"],
    data = [":testdata"],
    python_version = "PY3",
    tags = ["requires-net:external"],
    deps = [":gesture_recognizer_import"],
)
