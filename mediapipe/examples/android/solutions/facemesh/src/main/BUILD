# Copyright 2021 The MediaPipe Authors.
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
    name = "facemesh",
    srcs = glob(["**/*.java"]),
    custom_package = "com.google.mediapipe.examples.facemesh",
    manifest = "AndroidManifest.xml",
    manifest_values = {
        "applicationId": "com.google.mediapipe.examples.facemesh",
    },
    multidex = "native",
    resource_files = ["//mediapipe/examples/android/solutions:resource_files"],
    deps = [
        "//mediapipe/framework/formats:landmark_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/solutioncore:camera_input",
        "//mediapipe/java/com/google/mediapipe/solutioncore:mediapipe_jni_lib",
        "//mediapipe/java/com/google/mediapipe/solutioncore:solution_rendering",
        "//mediapipe/java/com/google/mediapipe/solutioncore:video_input",
        "//mediapipe/java/com/google/mediapipe/solutions/facemesh",
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
