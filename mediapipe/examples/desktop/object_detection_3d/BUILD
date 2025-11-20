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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

licenses(["notice"])

# bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/object_detection_3d:objectron_cpu
# To run 3D object detection for shoes,
# bazel-bin/mediapipe/examples/desktop/object_detection_3d/objectron_cpu \
#   --calculator_graph_config_file=mediapipe/graphs/object_detection_3d/objectron_desktop_cpu.pbtxt \
#   --input_side_packets="input_video_path=<input_video_path>,box_landmark_model_path=mediapipe/modules/objectron/object_detection_3d_sneakers.tflite,output_video_path=<output_video_path>,allowed_labels=Footwear"
# To detect objects from other categories, change box_landmark_model_path and allowed_labels accordingly.
# Chair: box_landmark_model_path=mediapipe/modules/objectron/object_detection_3d_chair.tflite,allowed_labels=Chair
# Camera: box_landmark_model_path=mediapipe/modules/objectron/object_detection_3d_camera.tflite,allowed_labels=Camera
# Cup: box_landmark_model_path=mediapipe/modules/objectron/object_detection_3d_cup.tflite,allowed_labels=Mug
cc_binary(
    name = "objectron_cpu",
    data = [
        "//mediapipe/modules/objectron:object_detection_3d_camera.tflite",
        "//mediapipe/modules/objectron:object_detection_3d_chair.tflite",
        "//mediapipe/modules/objectron:object_detection_3d_chair_1stage.tflite",
        "//mediapipe/modules/objectron:object_detection_3d_cup.tflite",
        "//mediapipe/modules/objectron:object_detection_3d_sneakers.tflite",
        "//mediapipe/modules/objectron:object_detection_3d_sneakers_1stage.tflite",
        "//mediapipe/modules/objectron:object_detection_ssd_mobilenetv2_oidv4_fp16.tflite",
    ],
    deps = [
        "//mediapipe/examples/desktop:simple_run_graph_main",
        "//mediapipe/graphs/object_detection_3d:desktop_cpu_calculators",
    ],
)
