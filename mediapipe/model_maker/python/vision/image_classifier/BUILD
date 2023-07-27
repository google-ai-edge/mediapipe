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
# Placeholder for internal Python library rule.

licenses(["notice"])

package(default_visibility = ["//mediapipe:__subpackages__"])

######################################################################
# Public target of the MediaPipe Model Maker ImageClassifier APIs.

# Please see https://developers.google.com/mediapipe/solutions/vision/image_classifier/customize for
# more information about the MediaPipe Model Maker ImageClassifier APIs.
######################################################################
py_library(
    name = "image_classifier_import",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":dataset",
        ":hyperparameters",
        ":image_classifier",
        ":image_classifier_options",
        ":model_options",
        ":model_spec",
    ],
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
    deps = [
        ":dataset",
        "//mediapipe/model_maker/python/vision/core:image_utils",
        "//mediapipe/model_maker/python/vision/core:test_utils",
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
    name = "image_classifier_options",
    srcs = ["image_classifier_options.py"],
    deps = [
        ":hyperparameters",
        ":model_options",
        ":model_spec",
    ],
)

py_library(
    name = "image_classifier",
    srcs = ["image_classifier.py"],
    deps = [
        ":hyperparameters",
        ":image_classifier_options",
        ":model_options",
        ":model_spec",
        "//mediapipe/model_maker/python/core/data:classification_dataset",
        "//mediapipe/model_maker/python/core/tasks:classifier",
        "//mediapipe/model_maker/python/core/utils:model_util",
        "//mediapipe/model_maker/python/core/utils:quantization",
        "//mediapipe/model_maker/python/vision/core:image_preprocessing",
        "//mediapipe/tasks/python/metadata/metadata_writers:image_classifier",
        "//mediapipe/tasks/python/metadata/metadata_writers:metadata_writer",
    ],
)

py_library(
    name = "image_classifier_test_lib",
    testonly = 1,
    srcs = ["image_classifier_test.py"],
    data = ["//mediapipe/model_maker/python/vision/image_classifier/testdata"],
    deps = [
        ":hyperparameters",
        ":image_classifier_import",
        ":model_options",
        "//mediapipe/tasks/python/test:test_utils",
    ],
)

py_test(
    name = "image_classifier_test",
    size = "large",
    srcs = ["image_classifier_test.py"],
    shard_count = 2,
    tags = ["requires-net:external"],
    deps = [
        ":image_classifier_test_lib",
    ],
)

py_binary(
    name = "image_classifier_demo",
    srcs = ["image_classifier_demo.py"],
    deps = [
        ":image_classifier_import",
        "//mediapipe/model_maker/python/core/utils:quantization",
    ],
)
