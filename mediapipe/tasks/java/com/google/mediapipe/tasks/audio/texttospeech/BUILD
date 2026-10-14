# Copyright 2026 The MediaPipe Authors.
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

load("@rules_kotlin//kotlin:android.bzl", "kt_android_library")  # Rewritten to @rules_kotlin by copybara.
load("//mediapipe/tasks/java/com/google/mediapipe/tasks:mediapipe_tasks_aar.bzl", "mediapipe_build_aar_with_jni", "mediapipe_jni_binary")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_jni_binary(
    name = "mediapipe_tasks_text_to_speech_jni",
    uses_explicit_exports = False,
    version_script = "//mediapipe/tasks/java:text_to_speech_version_script.lds",
    deps = [],
)

kt_android_library(
    name = "texttospeech_kt",
    srcs = [
        "MediaPipeTtsJni.kt",
        "TextToSpeech.kt",
    ],
    manifest = "AndroidManifest.xml",
    deps = [
        "//mediapipe/calculators/tensor:inference_calculator_java_proto_lite",
        "//mediapipe/tasks/cc/core/proto:acceleration_java_proto_lite",
        "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
        "//mediapipe/tasks/cc/core/proto:external_file_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
        "//third_party/kotlin/kotlinx_coroutines:kotlinx_coroutines-android",
    ],
)

android_library(
    name = "texttospeech",
    manifest = "AndroidManifest.xml",
    exports = [
        ":texttospeech_kt",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
    ],
    # The JNI library is attached here rather than to `texttospeech_kt`, as the OSS
    # `@rules_kotlin` `kt_android_library` only accepts Java/Kotlin deps.
    deps = [
        ":libmediapipe_tasks_text_to_speech_jni_lib",
        "@maven//:com_google_guava_guava",
    ],
)

mediapipe_build_aar_with_jni(
    name = "tasks_texttospeech",
    android_library = ":texttospeech",
    # The classes.jar of the ":texttospeech" AAR is empty because the Kotlin classes are only reachable
    # through `exports`; merge the compiled Kotlin jar explicitly.
    extra_classes_jars = [":texttospeech_kt_kt.jar"],
)
