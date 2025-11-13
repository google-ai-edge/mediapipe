# Copyright 2025 The MediaPipe Authors.
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

# Placeholder: load py_test

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

py_test(
    name = "serial_dispatcher_test",
    srcs = ["serial_dispatcher_test.py"],
    deps = [
        "//mediapipe/tasks/python/core:mediapipe_c_utils",
        "//mediapipe/tasks/python/core:serial_dispatcher",
        "//testing/pybase",
        "@mediapipe_pip_deps_absl_py//:pkg",
    ],
)

py_test(
    name = "mediapipe_c_bindings_test",
    srcs = ["mediapipe_c_bindings_test.py"],
    deps = [
        "//mediapipe/tasks/python/core:mediapipe_c_bindings",
        "//testing/pybase",
        "//testing/pybase:parameterized",
        "@mediapipe_pip_deps_absl_py//:pkg",
    ],
)

py_test(
    name = "async_result_dispatcher_test",
    srcs = ["async_result_dispatcher_test.py"],
    deps = [
        "//mediapipe/tasks/python/core:async_result_dispatcher",
        "//mediapipe/tasks/python/core:mediapipe_c_bindings",
        "@mediapipe_pip_deps_absl_py//:pkg",
    ],
)
