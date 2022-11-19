# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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

package(
    default_visibility = ["//mediapipe:__subpackages__"],
)

# TODO: Remove the unncessary test data once the demo data are moved to an open-sourced
# directory.
filegroup(
    name = "test_data",
    srcs = glob([
        "test_data/**",
    ]),
)

py_library(
    name = "constants",
    srcs = ["constants.py"],
)

# TODO: Change to py_library after migrating the MediaPipe hand solution
# library to MediaPipe hand task library.
py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    deps = [
        ":constants",
        "//mediapipe/model_maker/python/core/data:classification_dataset",
        "//mediapipe/model_maker/python/core/data:data_util",
        "//mediapipe/model_maker/python/core/utils:model_util",
        "//mediapipe/python/solutions:hands",
    ],
)

py_test(
    name = "dataset_test",
    srcs = ["dataset_test.py"],
    data = [
        ":test_data",
        "//mediapipe/model_maker/models/gesture_recognizer:models",
    ],
    deps = [
        ":dataset",
        "//mediapipe/python/solutions:hands",
        "//mediapipe/tasks/python/test:test_utils",
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
    data = ["//mediapipe/model_maker/models/gesture_recognizer:models"],
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
        "//mediapipe/tasks/python/metadata/metadata_writers:writer_utils",
    ],
)

py_test(
    name = "gesture_recognizer_test",
    size = "large",
    srcs = ["gesture_recognizer_test.py"],
    data = [
        ":test_data",
        "//mediapipe/model_maker/models/gesture_recognizer:models",
    ],
    shard_count = 2,
    deps = [
        ":gesture_recognizer_import",
        "//mediapipe/model_maker/python/core/utils:test_util",
        "//mediapipe/tasks/python/test:test_utils",
    ],
)

py_test(
    name = "metadata_writer_test",
    srcs = ["metadata_writer_test.py"],
    data = [
        ":test_data",
    ],
    deps = [
        ":metadata_writer",
        "//mediapipe/tasks/python/metadata/metadata_writers:metadata_writer",
        "//mediapipe/tasks/python/test:test_utils",
    ],
)

py_binary(
    name = "gesture_recognizer_demo",
    srcs = ["gesture_recognizer_demo.py"],
    data = [
        ":test_data",
        "//mediapipe/model_maker/models/gesture_recognizer:models",
    ],
    python_version = "PY3",
    deps = [":gesture_recognizer_import"],
)
