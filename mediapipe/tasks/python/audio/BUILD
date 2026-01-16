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

# Placeholder for internal Python strict library and test compatibility macro.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "audio_classifier",
    srcs = [
        "audio_classifier.py",
    ],
    deps = [
        "//mediapipe/tasks/python/audio/core:audio_record",
        "//mediapipe/tasks/python/audio/core:audio_task_running_mode",
        "//mediapipe/tasks/python/components/containers:audio_data",
        "//mediapipe/tasks/python/components/containers:audio_data_c",
        "//mediapipe/tasks/python/components/containers:classification_result",
        "//mediapipe/tasks/python/components/containers:classification_result_c",
        "//mediapipe/tasks/python/components/processors:classifier_options",
        "//mediapipe/tasks/python/components/processors:classifier_options_c",
        "//mediapipe/tasks/python/core:async_result_dispatcher",
        "//mediapipe/tasks/python/core:base_options",
        "//mediapipe/tasks/python/core:base_options_c",
        "//mediapipe/tasks/python/core:mediapipe_c_bindings",
        "//mediapipe/tasks/python/core:mediapipe_c_utils",
        "//mediapipe/tasks/python/core:optional_dependencies",
        "//mediapipe/tasks/python/core:serial_dispatcher",
    ],
)
