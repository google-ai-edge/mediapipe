# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
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

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

objc_library(
    name = "MPPCategoryHelpers",
    srcs = ["sources/MPPCategory+Helpers.mm"],
    hdrs = ["sources/MPPCategory+Helpers.h"],
    deps = [
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/components/containers:MPPCategory",
    ],
)

objc_library(
    name = "MPPClassificationResultHelpers",
    srcs = ["sources/MPPClassificationResult+Helpers.mm"],
    hdrs = ["sources/MPPClassificationResult+Helpers.h"],
    deps = [
        ":MPPCategoryHelpers",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_cc_proto",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/components/containers:MPPClassificationResult",
    ],
)
