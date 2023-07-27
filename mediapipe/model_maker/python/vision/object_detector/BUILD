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

# Placeholder for internal Python strict binary and library compatibility macro.
# Placeholder for internal Python strict test compatibility macro.

licenses(["notice"])

package(default_visibility = ["//mediapipe:__subpackages__"])

py_library(
    name = "object_detector_import",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":dataset",
        ":hyperparameters",
        ":model_options",
        ":model_spec",
        ":object_detector",
        ":object_detector_options",
    ],
)

py_binary(
    name = "object_detector_demo",
    srcs = ["object_detector_demo.py"],
    data = [":testdata"],
    python_version = "PY3",
    tags = ["requires-net:external"],
    deps = [":object_detector_import"],
)

filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/**",
    ]),
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    deps = [
        ":dataset_util",
        "//mediapipe/model_maker/python/core/data:cache_files",
        "//mediapipe/model_maker/python/core/data:classification_dataset",
    ],
)

py_test(
    name = "dataset_test",
    srcs = ["dataset_test.py"],
    data = [":testdata"],
    deps = [
        ":dataset",
        "//mediapipe/model_maker/python/vision/core:image_utils",
        "//mediapipe/model_maker/python/vision/core:test_utils",
        "//mediapipe/tasks/python/test:test_utils",
    ],
)

py_library(
    name = "dataset_util",
    srcs = ["dataset_util.py"],
    deps = ["//mediapipe/model_maker/python/core/data:cache_files"],
)

py_test(
    name = "dataset_util_test",
    srcs = ["dataset_util_test.py"],
    data = [":testdata"],
    deps = [
        ":dataset_util",
        "//mediapipe/model_maker/python/vision/core:test_utils",
        "//mediapipe/tasks/python/test:test_utils",
    ],
)

py_library(
    name = "detection",
    srcs = ["detection.py"],
)

py_test(
    name = "detection_test",
    srcs = ["detection_test.py"],
    deps = [":detection"],
)

py_library(
    name = "hyperparameters",
    srcs = ["hyperparameters.py"],
    deps = [
        "//mediapipe/model_maker/python/core:hyperparameters",
    ],
)

py_library(
    name = "preprocessor",
    srcs = ["preprocessor.py"],
    deps = [":model_spec"],
)

py_test(
    name = "preprocessor_test",
    srcs = ["preprocessor_test.py"],
    deps = [
        ":model_spec",
        ":preprocessor",
        "//mediapipe/model_maker/python/vision/core:test_utils",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        ":detection",
        ":model_options",
        ":model_spec",
    ],
)

py_test(
    name = "model_test",
    size = "large",
    srcs = ["model_test.py"],
    data = [":testdata"],
    shard_count = 4,
    tags = ["requires-net:external"],
    deps = [
        ":dataset",
        ":model",
        ":model_options",
        ":model_spec",
        ":preprocessor",
        "//mediapipe/tasks/python/test:test_utils",
    ],
)

py_library(
    name = "model_options",
    srcs = ["model_options.py"],
)

py_library(
    name = "model_spec",
    srcs = ["model_spec.py"],
    deps = ["//mediapipe/model_maker/python/core/utils:file_util"],
)

py_library(
    name = "object_detector",
    srcs = ["object_detector.py"],
    deps = [
        ":dataset",
        ":hyperparameters",
        ":model",
        ":model_options",
        ":model_spec",
        ":object_detector_options",
        ":preprocessor",
        "//mediapipe/model_maker/python/core/tasks:classifier",
        "//mediapipe/model_maker/python/core/utils:model_util",
        "//mediapipe/model_maker/python/core/utils:quantization",
        "//mediapipe/tasks/python/metadata/metadata_writers:metadata_info",
        "//mediapipe/tasks/python/metadata/metadata_writers:metadata_writer",
        "//mediapipe/tasks/python/metadata/metadata_writers:object_detector",
    ],
)

py_test(
    name = "object_detector_test",
    size = "enormous",
    srcs = ["object_detector_test.py"],
    data = [":testdata"],
    tags = ["requires-net:external"],
    deps = [
        ":object_detector_import",
        "//mediapipe/tasks/python/test:test_utils",
    ],
)

py_library(
    name = "object_detector_options",
    srcs = ["object_detector_options.py"],
    deps = [
        ":hyperparameters",
        ":model_options",
        ":model_spec",
    ],
)
