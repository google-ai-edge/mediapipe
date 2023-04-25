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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

android_library(
    name = "classifieroptions",
    srcs = ["ClassifierOptions.java"],
    javacopts = [
        "-Xep:AndroidJdkLibsChecker:OFF",
    ],
    deps = [
        "//mediapipe/tasks/cc/components/processors/proto:classifier_options_java_proto_lite",
        "//third_party:autovalue",
        "@maven//:com_google_guava_guava",
    ],
)

# Expose the java source files for building mediapipe tasks core AAR.
filegroup(
    name = "java_src",
    srcs = glob(["*.java"]),
    visibility = ["//mediapipe/tasks/java/com/google/mediapipe/tasks/core:__subpackages__"],
)
