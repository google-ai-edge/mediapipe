# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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

load(
    "//mediapipe/framework/tool:mediapipe_files.bzl",
    "mediapipe_files",
)

package(
    default_visibility = ["//mediapipe/framework:mediapipe_internal"],
    licenses = ["notice"],  # Apache 2.0
)

mediapipe_files(srcs = [
    "30k-clean.model",
    "albert_with_metadata.tflite",
    "bert_nl_classifier.tflite",
    "mobilebert_with_metadata.tflite",
    "test_model_nl_classifier_with_regex_tokenizer.tflite",
])

exports_files(srcs = [
    "mobilebert_vocab.txt",
    "vocab.txt",
    "vocab_with_index.txt",
])

filegroup(
    name = "vocab_files",
    srcs = [
        "empty_vocab_for_regex_tokenizer.txt",
        "mobilebert_vocab.txt",
        "vocab.txt",
        "vocab_for_regex_tokenizer.txt",
        "vocab_with_index.txt",
    ],
)

filegroup(
    name = "regex_tokenizer_files",
    srcs = [
        "empty_vocab_for_regex_tokenizer.txt",
        "vocab_for_regex_tokenizer.txt",
    ],
)

filegroup(
    name = "albert_model",
    srcs = [
        "30k-clean.model",
        "albert_with_metadata.tflite",
    ],
)

filegroup(
    name = "mobile_bert_model",
    srcs = [
        "mobilebert_vocab.txt",
        "mobilebert_with_metadata.tflite",
    ],
)

filegroup(
    name = "nl_classifier_models",
    srcs = glob([
        "test_model_nl_classifier*.tflite",
    ]),
)

filegroup(
    name = "bert_nl_classifier_models",
    srcs = ["bert_nl_classifier.tflite"],
)
