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

load("@rules_kotlin//kotlin:jvm.bzl", "kt_jvm_test")

licenses(["notice"])

kt_jvm_test(
    name = "ConfigParserTest",
    srcs = ["ConfigParserTest.kt"],
    deps = [
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/vision/generator",
        "//third_party/java/gson",
        "//third_party/java/junit",
        "//third_party/java/truth",
        "//third_party/kotlin/kotlin:kotlin_test",
    ],
)
