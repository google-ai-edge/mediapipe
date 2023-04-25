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

licenses(["notice"])

package(default_visibility = ["//visibility:private"])

android_binary(
    name = "objectdetector",
    srcs = glob(["**/*.java"]),
    assets = [
        "//mediapipe/tasks/testdata/vision:test_models",
    ],
    assets_dir = "",
    custom_package = "com.google.mediapipe.tasks.examples.objectdetector",
    manifest = "AndroidManifest.xml",
    manifest_values = {
        "applicationId": "com.google.mediapipe.tasks.examples.objectdetector",
    },
    multidex = "native",
    resource_files = ["//mediapipe/tasks/examples/android:resource_files"],
    deps = [
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/framework/image",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:detection",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:core",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:objectdetector",
        "//third_party:androidx_appcompat",
        "//third_party:androidx_constraint_layout",
        "//third_party:opencv",
        "@maven//:androidx_activity_activity",
        "@maven//:androidx_concurrent_concurrent_futures",
        "@maven//:androidx_exifinterface_exifinterface",
        "@maven//:androidx_fragment_fragment",
        "@maven//:com_google_guava_guava",
    ],
)
