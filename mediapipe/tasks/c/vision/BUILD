# Copyright 2023 The MediaPipe Authors.
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

VISION_LIBRARIES = [
    "//mediapipe/tasks/c/vision/hand_landmarker:hand_landmarker_lib",
    "//mediapipe/tasks/c/vision/image_classifier:image_classifier_lib",
    "//mediapipe/tasks/c/vision/image_embedder:image_embedder_lib",
    "//mediapipe/tasks/c/vision/object_detector:object_detector_lib",
]

# bazel build -c opt --linkopt -s --strip always --define MEDIAPIPE_DISABLE_GPU=1 \
# //mediapipe/tasks/c/vision:libvision.so
cc_binary(
    name = "libvision.so",
    linkopts = [
        "-Wl,-soname=libvision.so",
        "-fvisibility=hidden",
    ],
    linkshared = True,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = VISION_LIBRARIES,
)

# bazel build --config darwin_arm64 -c opt --strip always --define MEDIAPIPE_DISABLE_GPU=1 \
# //mediapipe/tasks/c/vision:libvision:.dylib
cc_binary(
    name = "libvision.dylib",
    linkopts = [
        "-Wl,-install_name,libvision.dylib",
        "-fvisibility=hidden",
    ],
    linkshared = True,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = VISION_LIBRARIES,
)
