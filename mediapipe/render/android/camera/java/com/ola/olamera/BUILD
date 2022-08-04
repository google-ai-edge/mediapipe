# Copyright 2019 The MediaPipe Authors.
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

exports_files([
    "proguard.pgcfg",
])

android_library(
    name = "camera_framework",
    # proguard_specs = [":proguard.pgcfg"],
    visibility = ["//visibility:public"],    
    exports = [
        ":camera_core",
    ],
)

android_library(
    name = "camera_core",
    srcs = glob(
        ["**/*.java"],
    ),
    # assets = [
    #     "//mediapipe/render/android/camera/java/com/ola/olamera:rbga_2_yuv_frag.frag",
    #     "//mediapipe/render/android/camera/java/com/ola/olamera:rgba_2_yuv_vertex.vert",
    # ],
    # assets_dir = "",
    # resource_files = glob(["res/**"]),
    javacopts = ["-Acom.google.auto.value.AutoBuilderIsUnstable"],
    # manifest = ":AndroidManifest.xml",
    visibility = ["//visibility:public"],
    deps = [
        "@maven//:com_google_guava_guava",
        "//third_party:androidx_annotation",
        "//third_party:androidx_appcompat",
        "//third_party:camerax_camera2",
        "//third_party:camerax_core",
        "//third_party:camerax_lifecycle",
        "//third_party:lifecycle_livedata",
        "//third_party:lifecycle_livedata_core",
        "//third_party:androidx_core",
        "//third_party:androidx_lifecycle_common",
        "//third_party:androidx_lifecycle_runtime",
        "@maven//:androidx_exifinterface_exifinterface",
    ],
)