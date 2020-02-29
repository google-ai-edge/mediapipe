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

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "asset_manager_util",
    srcs = select({
        "//conditions:default": [],  # don't build other than android.
        "//mediapipe:android": [
            "asset_manager_util.cc",
        ],
    }),
    hdrs = ["asset_manager_util.h"],
    linkopts = select({
        "//conditions:default": [],
        "//mediapipe:android": ["-landroid"],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:singleton",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "//mediapipe/util/android/file/base",
    ] + select({
        "//conditions:default": [],
        "//mediapipe:android": [":jni_helper"],
    }),
)

cc_library(
    name = "jni_helper",
    srcs = select({
        "//conditions:default": [],
        "//mediapipe:android": ["jni_helper.cc"],
    }),
    hdrs = select({
        "//conditions:default": [],
        "//mediapipe:android": ["jni_helper.h"],
    }),
    linkopts = ["-landroid"],
    deps = [":logging"],
)

cc_library(
    name = "logging",
    hdrs = select({
        "//conditions:default": [],
        "//mediapipe:android": ["logging.h"],
    }),
    linkopts = [
        "-landroid",
        "-llog",
    ],
)
