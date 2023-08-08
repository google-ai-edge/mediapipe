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
    data = [
        "//mediapipe/models:ssdlite_object_detection.tflite",
        "//mediapipe/models:ssdlite_object_detection_labelmap.txt",
    ],
    deps = [
        "//mediapipe/examples/desktop:simple_run_graph_main",
        "//mediapipe/graphs/object_detection:desktop_tensorflow_calculators",
        "@org_tensorflow//tensorflow/core:all_kernels",
        "@org_tensorflow//tensorflow/core:direct_session",
    ],
)

cc_binary(
    name = "object_detection_tflite",
    data = [
        "//mediapipe/models:ssdlite_object_detection.tflite",
        "//mediapipe/models:ssdlite_object_detection_labelmap.txt",
    ],
    deps = [
        "//mediapipe/examples/desktop:simple_run_graph_main",
        "//mediapipe/graphs/object_detection:desktop_tflite_calculators",
    ],
)

cc_binary(
    name = "object_detection_openvino",
    data = [
        "//mediapipe/models/ssdlite_object_detection_openvino/1:ssdlite_object_detection.xml",
        "//mediapipe/models/ssdlite_object_detection_openvino/1:ssdlite_object_detection.bin",
        "//mediapipe/models:ssdlite_object_detection_labelmap.txt",
    ],
    deps = [
        "//mediapipe/examples/desktop:simple_run_graph_main",
        "//mediapipe/graphs/object_detection:desktop_openvino_calculators",
    ],
)

cc_binary(
    name = "object_detection_ovms",
    data = [
        "//mediapipe/models/ssdlite_object_detection_openvino/1:ssdlite_object_detection.xml",
        "//mediapipe/models/ssdlite_object_detection_openvino/1:ssdlite_object_detection.bin",
        "//mediapipe/models:ssdlite_object_detection_labelmap.txt",
	],
    deps = [
        "//mediapipe/examples/desktop:simple_run_graph_main",
        "//mediapipe/graphs/object_detection:desktop_ovms_calculators",
        # here we declare dependencies on tflite as we can support 2 types
        # of graphs:
        # 1) using tflite converters & tensors as packet streams
        # 2) using openvino converters & tensors as packet streams
        "//mediapipe/graphs/object_detection:desktop_tflite_calculators",
    ],
)

cc_binary(
    name = "object_detection_cpu",
    data = [
        "//mediapipe/models:ssdlite_object_detection.tflite",
        "//mediapipe/models:ssdlite_object_detection_labelmap.txt",
    ],
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main",
        "//mediapipe/graphs/object_detection:desktop_tflite_calculators",
    ],
)

