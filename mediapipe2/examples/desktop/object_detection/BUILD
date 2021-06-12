# Copyright 2019 The MediaPipe Authors.
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

licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

cc_binary(
    name = "object_detection_tensorflow",
    deps = [
        "//mediapipe/examples/desktop:simple_run_graph_main",
        "//mediapipe/graphs/object_detection:desktop_tensorflow_calculators",
        "@org_tensorflow//tensorflow/core:all_kernels",
        "@org_tensorflow//tensorflow/core:direct_session",
    ],
)

cc_binary(
    name = "object_detection_tflite",
    deps = [
        "//mediapipe/examples/desktop:simple_run_graph_main",
        "//mediapipe/graphs/object_detection:desktop_tflite_calculators",
    ],
)

cc_binary(
    name = "object_detection_cpu",
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main",
        "//mediapipe/graphs/object_detection:desktop_tflite_calculators",
    ],
)
