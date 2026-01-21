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

licenses(["notice"])

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
        "@mediapipe_pip_deps_absl_py//:pkg",
    ],
)
