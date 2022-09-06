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

load(
    "//mediapipe/framework/tool:mediapipe_files.bzl",
    "mediapipe_files",
)

package(
    default_visibility = ["//mediapipe/tasks:internal"],
    licenses = ["notice"],  # Apache 2.0
)

mediapipe_files(srcs = [
    "corrupted_mobilenet_v1_0.25_224_1_default_1.tflite",
    "mobilenet_v1_0.25_224_quant.tflite",
    "test_model_add_op.tflite",
    "test_model_with_custom_op.tflite",
    "test_model_without_custom_op.tflite",
])

exports_files(["external_file.txt"])

filegroup(
    name = "test_models",
    srcs = [
        "corrupted_mobilenet_v1_0.25_224_1_default_1.tflite",
        "mobilenet_v1_0.25_224_quant.tflite",
        "test_model_add_op.tflite",
        "test_model_with_custom_op.tflite",
        "test_model_without_custom_op.tflite",
    ],
)

filegroup(
    name = "external_file",
    srcs = ["external_file.txt"],
)
