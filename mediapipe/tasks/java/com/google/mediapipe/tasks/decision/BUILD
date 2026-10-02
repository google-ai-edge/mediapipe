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
    name = "mediapipe_tasks_decision_jni",
    # Hide all non-JNI symbols. `-fvisibility=hidden` alone is not enough as it
    # only applies to the sources of the JNI binary itself, not to transitive
    # deps such as the LiteRT-LM protos.
    uses_explicit_exports = False,
    version_script = "//mediapipe/tasks/java:decision_version_script.lds",
    deps = [],
)

kt_android_library(
    name = "decision_kt",
    srcs = [
        "DecisionMaker.kt",
        "DecisionMakerOptions.kt",
        "DecisionQuestion.kt",
        "DecisionResult.kt",
    ],
    manifest = "AndroidManifest.xml",
    deps = [
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
    ],
)

android_library(
    name = "decision",
    manifest = "AndroidManifest.xml",
    exports = [":decision_kt"],
    # The JNI library is attached here rather than to `decision_kt`, as the OSS
    # `@rules_kotlin` `kt_android_library` only accepts Java/Kotlin deps.
    deps = [
        ":libmediapipe_tasks_decision_jni_lib",
        "@maven//:com_google_guava_guava",
    ],
)

mediapipe_build_aar_with_jni(
    name = "tasks_decision",
    android_library = ":decision",
    # The classes.jar of the ":decision" AAR is empty because the Kotlin classes are only reachable
    # through `exports`; merge the compiled Kotlin jar explicitly.
    extra_classes_jars = [":decision_kt_kt.jar"],
)
