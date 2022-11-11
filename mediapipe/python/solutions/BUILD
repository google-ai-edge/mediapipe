# Copyright 2020 The MediaPipe Authors.
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

package(default_visibility = ["//visibility:public"])

py_library(
    name = "hands",
    srcs = [
        "hands.py",
        "hands_connections.py",
    ],
    data = [
        "//mediapipe/modules/hand_landmark:hand_landmark_full.tflite",
        "//mediapipe/modules/hand_landmark:hand_landmark_lite.tflite",
        "//mediapipe/modules/hand_landmark:hand_landmark_tracking_cpu_graph",
        "//mediapipe/modules/hand_landmark:handedness.txt",
        "//mediapipe/modules/palm_detection:palm_detection_full.tflite",
        "//mediapipe/modules/palm_detection:palm_detection_lite.tflite",
    ],
    srcs_version = "PY3",
    deps = [
        "//mediapipe/calculators/core:constant_side_packet_calculator_py_pb2",
        "//mediapipe/calculators/core:gate_calculator_py_pb2",
        "//mediapipe/calculators/core:split_vector_calculator_py_pb2",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator_py_pb2",
        "//mediapipe/calculators/tensor:inference_calculator_py_pb2",
        "//mediapipe/calculators/tensor:tensors_to_classification_calculator_py_pb2",
        "//mediapipe/calculators/tensor:tensors_to_detections_calculator_py_pb2",
        "//mediapipe/calculators/tensor:tensors_to_landmarks_calculator_py_pb2",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator_py_pb2",
        "//mediapipe/calculators/util:association_calculator_py_pb2",
        "//mediapipe/calculators/util:detections_to_rects_calculator_py_pb2",
        "//mediapipe/calculators/util:logic_calculator_py_pb2",
        "//mediapipe/calculators/util:non_max_suppression_calculator_py_pb2",
        "//mediapipe/calculators/util:rect_transformation_calculator_py_pb2",
        "//mediapipe/calculators/util:thresholding_calculator_py_pb2",
        "//mediapipe/python:solution_base",
    ],
)

py_library(
    name = "drawing_styles",
    srcs = ["drawing_styles.py"],
    srcs_version = "PY3",
    deps = [
        "drawing_utils",
        "face_mesh",
        "hands",
        "pose",
    ],
)

py_library(
    name = "drawing_utils",
    srcs = ["drawing_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//mediapipe/framework/formats:detection_py_pb2",
        "//mediapipe/framework/formats:landmark_py_pb2",
        "//mediapipe/framework/formats:location_data_py_pb2",
    ],
)

py_test(
    name = "drawing_utils_test",
    srcs = ["drawing_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":drawing_utils",
        "//mediapipe/framework/formats:detection_py_pb2",
        "//mediapipe/framework/formats:landmark_py_pb2",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_test(
    name = "hands_test",
    srcs = ["hands_test.py"],
    data = [
        ":testdata/asl_hand.25fps.mp4",
        ":testdata/asl_hand.full.npz",
        ":testdata/hands.jpg",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":drawing_styles",
        ":drawing_utils",
        ":hands",
    ],
)
