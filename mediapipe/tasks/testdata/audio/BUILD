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
    "model_without_metadata.tflite",
    "speech_16000_hz_mono.wav",
    "speech_48000_hz_mono.wav",
    "two_heads.tflite",
    "two_heads_16000_hz_mono.wav",
    "two_heads_44100_hz_mono.wav",
    "yamnet_audio_classifier_with_metadata.tflite",
    "yamnet_embedding_metadata.tflite",
])

filegroup(
    name = "test_models",
    srcs = [
        "model_without_metadata.tflite",
        "two_heads.tflite",
        "yamnet_audio_classifier_with_metadata.tflite",
        "yamnet_embedding_metadata.tflite",
    ],
)

filegroup(
    name = "test_audio_clips",
    srcs = [
        "speech_16000_hz_mono.wav",
        "speech_48000_hz_mono.wav",
        "two_heads_16000_hz_mono.wav",
        "two_heads_44100_hz_mono.wav",
    ],
)
