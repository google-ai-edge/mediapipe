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

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

TEXT_LIBRARIES = [
    "//mediapipe/tasks/c/text/language_detector:language_detector_lib",
    "//mediapipe/tasks/c/text/text_classifier:text_classifier_lib",
    "//mediapipe/tasks/c/text/text_embedder:text_embedder_lib",
]

# bazel build -c opt --linkopt -s --strip always --define MEDIAPIPE_DISABLE_GPU=1 \
# //mediapipe/tasks/c/text:libtext.so
cc_binary(
    name = "libtext.so",
    linkopts = [
        "-Wl,-soname=libtext.so",
        "-fvisibility=hidden",
    ],
    linkshared = True,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = TEXT_LIBRARIES,
)

# bazel build --config darwin_arm64 -c opt --strip always --define MEDIAPIPE_DISABLE_GPU=1 \
# //mediapipe/tasks/c/text:libtext:.dylib
cc_binary(
    name = "libtext.dylib",
    linkopts = [
        "-Wl,-install_name,libtext.dylib",
        "-fvisibility=hidden",
    ],
    linkshared = True,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = TEXT_LIBRARIES,
)
