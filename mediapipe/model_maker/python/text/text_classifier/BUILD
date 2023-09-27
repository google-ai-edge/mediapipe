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

package(default_visibility = ["//mediapipe:__subpackages__"])

licenses(["notice"])

######################################################################
# Public target of the MediaPipe Model Maker TextCassifier APIs.

# Please see https://developers.google.com/mediapipe/solutions/text/text_classifier/customize for
# more information about the MediaPipe Model Maker TextCassifier APIs.
######################################################################
py_library(
    name = "text_classifier_import",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":bert_tokenizer",
        ":dataset",
        ":hyperparameters",
        ":model_options",
        ":model_spec",
        ":preprocessor",
        ":text_classifier",
        ":text_classifier_options",
    ],
)

py_library(
    name = "model_options",
    srcs = ["model_options.py"],
    deps = ["//mediapipe/model_maker/python/text/core:bert_model_options"],
)

py_library(
    name = "hyperparameters",
    srcs = ["hyperparameters.py"],
    deps = [
        ":bert_tokenizer",
        "//mediapipe/model_maker/python/core:hyperparameters",
    ],
)

py_library(
    name = "model_spec",
    srcs = ["model_spec.py"],
    deps = [
        ":hyperparameters",
        ":model_options",
        "//mediapipe/model_maker/python/core/utils:file_util",
        "//mediapipe/model_maker/python/text/core:bert_model_spec",
    ],
)

py_test(
    name = "model_spec_test",
    srcs = ["model_spec_test.py"],
    tags = ["requires-net:external"],
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
        "//mediapipe/model_maker/python/core/data:cache_files",
        "//mediapipe/model_maker/python/core/data:classification_dataset",
    ],
)

py_test(
    name = "dataset_test",
    srcs = ["dataset_test.py"],
    deps = [":dataset"],
)

py_library(
    name = "model_with_tokenizer",
    srcs = ["model_with_tokenizer.py"],
)

py_test(
    name = "model_with_tokenizer_test",
    srcs = ["model_with_tokenizer_test.py"],
    tags = ["requires-net:external"],
    deps = [
        ":bert_tokenizer",
        ":model_spec",
        ":model_with_tokenizer",
        "//mediapipe/model_maker/python/core/utils:hub_loader",
    ],
)

py_library(
    name = "bert_tokenizer",
    srcs = ["bert_tokenizer.py"],
)

py_test(
    name = "bert_tokenizer_test",
    srcs = ["bert_tokenizer_test.py"],
    tags = ["requires-net:external"],
    deps = [
        ":bert_tokenizer",
        ":model_spec",
    ],
)

py_library(
    name = "preprocessor",
    srcs = ["preprocessor.py"],
    deps = [
        ":bert_tokenizer",
        ":dataset",
        "//mediapipe/model_maker/python/core/data:cache_files",
    ],
)

py_test(
    name = "preprocessor_test",
    srcs = ["preprocessor_test.py"],
    tags = ["requires-net:external"],
    deps = [
        ":bert_tokenizer",
        ":dataset",
        ":model_spec",
        ":preprocessor",
        "//mediapipe/model_maker/python/core/data:cache_files",
    ],
)

py_library(
    name = "text_classifier_options",
    srcs = ["text_classifier_options.py"],
    deps = [
        ":hyperparameters",
        ":model_options",
        ":model_spec",
    ],
)

py_library(
    name = "text_classifier",
    srcs = ["text_classifier.py"],
    deps = [
        ":bert_tokenizer",
        ":dataset",
        ":hyperparameters",
        ":model_options",
        ":model_spec",
        ":model_with_tokenizer",
        ":preprocessor",
        ":text_classifier_options",
        "//mediapipe/model_maker/python/core/data:dataset",
        "//mediapipe/model_maker/python/core/tasks:classifier",
        "//mediapipe/model_maker/python/core/utils:hub_loader",
        "//mediapipe/model_maker/python/core/utils:loss_functions",
        "//mediapipe/model_maker/python/core/utils:metrics",
        "//mediapipe/model_maker/python/core/utils:model_util",
        "//mediapipe/model_maker/python/core/utils:quantization",
        "//mediapipe/tasks/python/metadata/metadata_writers:metadata_writer",
        "//mediapipe/tasks/python/metadata/metadata_writers:text_classifier",
    ],
)

py_test(
    name = "text_classifier_test",
    size = "enormous",
    srcs = ["text_classifier_test.py"],
    data = [
        "//mediapipe/model_maker/python/text/text_classifier/testdata",
    ],
    tags = [
        "noasan",
        "notsan",
        "requires-mem:16g",
        "requires-net:external",
    ],
    deps = [
        ":text_classifier_import",
        "//mediapipe/model_maker/python/core/utils:loss_functions",
        "//mediapipe/tasks/python/test:test_utils",
    ],
)

py_library(
    name = "text_classifier_demo_lib",
    srcs = ["text_classifier_demo.py"],
    deps = [
        ":text_classifier_import",
        "//mediapipe/model_maker/python/core/utils:quantization",
    ],
)

py_binary(
    name = "text_classifier_demo",
    srcs = ["text_classifier_demo.py"],
    deps = [
        ":text_classifier_demo_lib",
    ],
)
