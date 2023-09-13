# Copyright 2022 The MediaPipe Authors.
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

# Placeholder for internal Python strict test compatibility macro.

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

py_test(
    name = "text_classifier_test",
    srcs = ["text_classifier_test.py"],
    data = [
        "//mediapipe/tasks/testdata/text:bert_text_classifier_models",
        "//mediapipe/tasks/testdata/text:text_classifier_models",
    ],
    tags = ["not_run:arm"],
    deps = [
        "//mediapipe/tasks/python/components/containers:category",
        "//mediapipe/tasks/python/components/containers:classification_result",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/test:test_utils",
        "//mediapipe/tasks/python/text:text_classifier",
    ],
)

py_test(
    name = "text_embedder_test",
    srcs = ["text_embedder_test.py"],
    data = [
        "//mediapipe/tasks/testdata/text:mobilebert_embedding_model",
        "//mediapipe/tasks/testdata/text:regex_embedding_with_metadata",
        "//mediapipe/tasks/testdata/text:universal_sentence_encoder_qa",
    ],
    tags = ["not_run:arm"],
    deps = [
        "//mediapipe/tasks/python/components/containers:embedding_result",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/test:test_utils",
        "//mediapipe/tasks/python/text:text_embedder",
    ],
)

py_test(
    name = "language_detector_test",
    srcs = ["language_detector_test.py"],
    data = [
        "//mediapipe/tasks/testdata/text:language_detector",
    ],
    deps = [
        "//mediapipe/tasks/python/components/containers:category",
        "//mediapipe/tasks/python/components/containers:classification_result",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/test:test_utils",
        "//mediapipe/tasks/python/text:language_detector",
    ],
)
