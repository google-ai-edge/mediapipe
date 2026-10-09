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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load(":mediapipe_tasks_c_shared_library.bzl", "mediapipe_tasks_c_shared_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

exports_files([
    "mediapipe_tasks_audio_version_script.lds",
    "mediapipe_tasks_c_version_script.lds",
    "mediapipe_tasks_retrieval_version_script.lds",
    "mediapipe_tasks_text_version_script.lds",
    "mediapipe_tasks_vision_version_script.lds",
])

# C API libraries that are part of every domain library.
MEDIAPIPE_TASKS_C_CORE_DEPS = [
    "//mediapipe/tasks/c/core:common",
    "//mediapipe/tasks/c/core:mp_logging_c_lib",
    "//mediapipe/tasks/c/metadata:flatbuffer_api_c_lib",
]

MEDIAPIPE_TASKS_C_VISION_DEPS = [
    "//mediapipe/tasks/c/vision/core:image",
    "//mediapipe/tasks/c/vision/face_detector:face_detector_c_lib",
    "//mediapipe/tasks/c/vision/face_landmarker:face_landmarker_c_lib",
    "//mediapipe/tasks/c/vision/gesture_recognizer:gesture_recognizer_c_lib",
    "//mediapipe/tasks/c/vision/hand_landmarker:hand_landmarker_c_lib",
    "//mediapipe/tasks/c/vision/holistic_landmarker:holistic_landmarker_c_lib",
    "//mediapipe/tasks/c/vision/image_classifier:image_classifier_c_lib",
    "//mediapipe/tasks/c/vision/image_embedder:image_embedder_c_lib",
    "//mediapipe/tasks/c/vision/image_segmenter:image_segmenter_c_lib",
    "//mediapipe/tasks/c/vision/interactive_segmenter_legacy:interactive_segmenter_legacy_c_lib",
    "//mediapipe/tasks/c/vision/object_detector:object_detector_c_lib",
    "//mediapipe/tasks/c/vision/pose_landmarker:pose_landmarker_c_lib",
]

MEDIAPIPE_TASKS_C_TEXT_DEPS = [
    "//mediapipe/tasks/c/text/language_detector:language_detector_c_lib",
    "//mediapipe/tasks/c/text/text_classifier:text_classifier_c_lib",
    "//mediapipe/tasks/c/text/text_embedder:text_embedder_c_lib",
]

MEDIAPIPE_TASKS_C_AUDIO_DEPS = [
    "//mediapipe/tasks/c/audio/audio_classifier:audio_classifier_c_lib",
    "//mediapipe/tasks/c/audio/audio_embedder:audio_embedder_c_lib",
]

MEDIAPIPE_TASKS_C_RETRIEVAL_DEPS = [
    "//mediapipe/tasks/c/retrieval/semantic_retriever:semantic_retriever_lib",
    "//mediapipe/tasks/c/retrieval/universal_embedder:universal_embedder_lib",
]

# Dependencies of the standalone retrieval library that are already provided by
# another domain in the monolithic library: the retrieval APIs consume MpImage
# handles and image/text embedders, so the standalone library carries its own
# copies of those C APIs, and the sqlite vector/memory stores only depend on the
# sqlite headers and expect the final binary to provide the implementation.
MEDIAPIPE_TASKS_C_RETRIEVAL_BORROWED_DEPS = [
    "//mediapipe/tasks/c/text/text_embedder:text_embedder_c_lib",
    "//mediapipe/tasks/c/vision/core:image",
    "//mediapipe/tasks/c/vision/image_embedder:image_embedder_c_lib",
]

# Per-domain shared libraries. Each library is self-contained and exports only
# the `Mp*` symbols of its domain (plus the core symbols), so that an
# application can load any subset of them. Handles (e.g. MpImage) must not be
# shared across libraries.
#
# Desktop / Android (see release/build_locally.sh for the full flag sets):
#   bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
#     //mediapipe/tasks/c:mediapipe_tasks_vision_lib
# iOS (device + simulator xcframework):
#   bazel build --config=ios -c opt --apple_generate_dsym=false \
#     //mediapipe/tasks/c:MediaPipeTasksVisionC
mediapipe_tasks_c_shared_library(
    name = "mediapipe_tasks_vision",
    framework_bundle_id = "com.google.mediapipe.tasks.c.vision",
    framework_name = "MediaPipeTasksVisionC",
    version_script = ":mediapipe_tasks_vision_version_script.lds",
    visibility = ["//visibility:public"],
    deps = MEDIAPIPE_TASKS_C_CORE_DEPS + MEDIAPIPE_TASKS_C_VISION_DEPS,
)

mediapipe_tasks_c_shared_library(
    name = "mediapipe_tasks_text",
    framework_bundle_id = "com.google.mediapipe.tasks.c.text",
    framework_name = "MediaPipeTasksTextC",
    version_script = ":mediapipe_tasks_text_version_script.lds",
    visibility = ["//visibility:public"],
    deps = MEDIAPIPE_TASKS_C_CORE_DEPS + MEDIAPIPE_TASKS_C_TEXT_DEPS,
)

mediapipe_tasks_c_shared_library(
    name = "mediapipe_tasks_audio",
    framework_bundle_id = "com.google.mediapipe.tasks.c.audio",
    framework_name = "MediaPipeTasksAudioC",
    version_script = ":mediapipe_tasks_audio_version_script.lds",
    visibility = ["//visibility:public"],
    deps = MEDIAPIPE_TASKS_C_CORE_DEPS + MEDIAPIPE_TASKS_C_AUDIO_DEPS,
)

mediapipe_tasks_c_shared_library(
    name = "mediapipe_tasks_retrieval",
    framework_bundle_id = "com.google.mediapipe.tasks.c.retrieval",
    framework_name = "MediaPipeTasksRetrievalC",
    version_script = ":mediapipe_tasks_retrieval_version_script.lds",
    visibility = ["//visibility:public"],
    deps = MEDIAPIPE_TASKS_C_CORE_DEPS +
           MEDIAPIPE_TASKS_C_RETRIEVAL_DEPS +
           MEDIAPIPE_TASKS_C_RETRIEVAL_BORROWED_DEPS,
)

# Monolithic library containing all domains. Used by the Python package.
#
# bazel build -c opt --linkopt -s --strip always --define MEDIAPIPE_DISABLE_GPU=1 \
# //mediapipe/tasks/c:libmediapipe.so
cc_binary(
    name = "mediapipe_source",
    linkopts = select({
        "@platforms//os:linux": [
            "-Wl,-soname=libmediapipe.so",
            "-Wl,--version-script=$(execpath :mediapipe_tasks_c_version_script.lds)",
            "-fvisibility=hidden",
        ],
        "//conditions:default": [],
    }),
    linkshared = True,
    linkstatic = True,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [":mediapipe_tasks_c_version_script.lds"] +
           MEDIAPIPE_TASKS_C_CORE_DEPS +
           MEDIAPIPE_TASKS_C_AUDIO_DEPS +
           MEDIAPIPE_TASKS_C_RETRIEVAL_DEPS +
           MEDIAPIPE_TASKS_C_TEXT_DEPS +
           MEDIAPIPE_TASKS_C_VISION_DEPS,
)

# Genrule to provide a consistent libmediapipe.so output in bazel-bin.
# buildifier: disable=constant-glob
genrule(
    name = "mediapipe_linux",
    srcs = glob(["prebuilts/manylinux/libmediapipe.so"]) or [":mediapipe_source"],
    outs = ["libmediapipe.so"],
    cmd = "cp $< $@",
)

# buildifier: disable=constant-glob
genrule(
    name = "mediapipe_windows",
    srcs = glob(["prebuilts/windows_x86_64/libmediapipe.dll"]) or [":mediapipe_source"],
    outs = ["libmediapipe.dll"],
    cmd = "cp $< $@",
)

# buildifier: disable=constant-glob
genrule(
    name = "mediapipe_macos",
    srcs = glob(["prebuilts/darwin_arm64/libmediapipe.dylib"]) or [":mediapipe_source"],
    outs = ["libmediapipe.dylib"],
    cmd = "cp $< $@",
)

alias(
    name = "libmediapipe",
    actual = select({
        "@platforms//os:windows": ":libmediapipe.dll",
        "@platforms//os:osx": ":libmediapipe.dylib",
        "//conditions:default": ":libmediapipe.so",
    }),
    visibility = ["//visibility:public"],
)
