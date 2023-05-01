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
    name = "audio_classifier_test",
    srcs = ["audio_classifier_test.py"],
    data = [
        "//mediapipe/tasks/testdata/audio:test_audio_clips",
        "//mediapipe/tasks/testdata/audio:test_models",
    ],
    deps = [
        "//mediapipe/tasks/python/audio:audio_classifier",
        "//mediapipe/tasks/python/audio/core:audio_record",
        "//mediapipe/tasks/python/audio/core:audio_task_running_mode",
        "//mediapipe/tasks/python/components/containers:audio_data",
        "//mediapipe/tasks/python/components/containers:classification_result",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/test:test_utils",
    ],
)

py_test(
    name = "audio_embedder_test",
    srcs = ["audio_embedder_test.py"],
    data = [
        "//mediapipe/tasks/testdata/audio:test_audio_clips",
        "//mediapipe/tasks/testdata/audio:test_models",
    ],
    deps = [
        "//mediapipe/tasks/python/audio:audio_embedder",
        "//mediapipe/tasks/python/audio/core:audio_record",
        "//mediapipe/tasks/python/audio/core:audio_task_running_mode",
        "//mediapipe/tasks/python/components/containers:audio_data",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/test:test_utils",
    ],
)
