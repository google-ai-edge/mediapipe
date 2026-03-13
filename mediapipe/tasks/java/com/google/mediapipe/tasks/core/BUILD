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

load("//mediapipe/tasks/java/com/google/mediapipe/tasks:mediapipe_tasks_aar.bzl", "mediapipe_jni_binary", "mediapipe_tasks_core_aar")

package(default_visibility = ["//visibility:public"])

android_library(
    name = "core",
    manifest = "AndroidManifest.xml",
    exports = [
        ":core_java",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core/jni:model_resources_cache_jni",
    ],
    deps = ["@maven//:com_google_guava_guava"],
)

mediapipe_jni_binary(
    name = "mediapipe_tasks_jni",
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/java/com/google/mediapipe/framework/jni:mediapipe_framework_jni",
        "//mediapipe/tasks/cc/audio/audio_classifier:audio_classifier_graph",
        "//mediapipe/tasks/cc/text/text_classifier:text_classifier_graph",
        "//mediapipe/tasks/cc/text/text_embedder:text_embedder_graph",
        "//mediapipe/tasks/cc/vision/face_detector:face_detector_graph",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker_graph",
        "//mediapipe/tasks/cc/vision/gesture_recognizer:gesture_recognizer_graph",
        "//mediapipe/tasks/cc/vision/holistic_landmarker:holistic_landmarker_graph",
        "//mediapipe/tasks/cc/vision/image_classifier:image_classifier_graph",
        "//mediapipe/tasks/cc/vision/image_embedder:image_embedder_graph",
        "//mediapipe/tasks/cc/vision/image_segmenter:image_segmenter_graph",
        "//mediapipe/tasks/cc/vision/interactive_segmenter:interactive_segmenter_graph",
        "//mediapipe/tasks/cc/vision/object_detector:object_detector_graph",
        "//mediapipe/tasks/cc/vision/pose_landmarker:pose_landmarker_graph",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core/jni:model_resources_cache_jni",
    ],
)

android_library(
    name = "core_java",
    srcs = glob(["*.java"]),
    manifest = "AndroidManifest.xml",
    deps = [
        ":logging",
        "//mediapipe/calculators/core:flow_limiter_calculator_java_proto_lite",
        "//mediapipe/calculators/tensor:inference_calculator_java_proto_lite",
        "//mediapipe/framework:calculator_java_proto_lite",
        "//mediapipe/framework:calculator_options_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework_no_mff",
        "//mediapipe/tasks/cc/core/proto:acceleration_java_proto_lite",
        "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
        "//mediapipe/tasks/cc/core/proto:external_file_java_proto_lite",
        "//third_party:any_java_proto",
        "//third_party:autovalue",
        "@com_google_protobuf//:protobuf_javalite",
        "@maven//:com_google_guava_guava",
    ],
)

# When "--define ENABLE_TASKS_USAGE_LOGGING=1" is set in the build command,
# the mediapipe tasks usage logging component will be added into the AAR.
# This flag is for internal use only.
config_setting(
    name = "enable_tasks_usage_logging",
    define_values = {
        "ENABLE_TASKS_USAGE_LOGGING": "1",
    },
    visibility = ["//visibility:public"],
)

genrule(
    name = "logging_factory",
    srcs = ["logging/TasksStatsDummyLogger.java"],
    outs = ["logging/TasksStatsLoggerFactory.java"],
    cmd = """
cat <<EOF > $@
package com.google.mediapipe.tasks.core.logging;
import android.content.Context;
public final class TasksStatsLoggerFactory {
  public static TasksStatsLogger create(
      Context context, String taskNameStr, String taskRunningModeStr) {
    return $$(basename $< .java).create(context, taskNameStr, taskRunningModeStr);
  }
}
EOF""",
)

filegroup(
    name = "logging_java_src",
    srcs = [
        "logging/TasksStatsDummyLogger.java",
        "logging/TasksStatsLogger.java",
        "logging/TasksStatsLoggerFactory.java",
    ],
)

android_library(
    name = "logging",
    srcs = [":logging_java_src"],
    deps = [
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

mediapipe_tasks_core_aar(
    name = "tasks_core",
    srcs = glob(["*.java"]) + [
        ":logging_java_src",
        "//mediapipe/java/com/google/mediapipe/framework/image:java_src",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:java_src",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/processors:java_src",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/utils:java_src",
    ],
    manifest = "AndroidManifest.xml",
    deps = [
        ":libmediapipe_tasks_jni_lib",
    ],
)
