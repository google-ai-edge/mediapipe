# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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
    name = "MPPTextClassifierOptionsHelpers",
    srcs = ["sources/MPPTextClassifierOptions+Helpers.mm"],
    hdrs = ["sources/MPPTextClassifierOptions+Helpers.h"],
    deps = [
        "//mediapipe/tasks/ios/text/text_classifier:MPPTextClassifierOptions",
        "//mediapipe/tasks/ios/core/utils:MPPBaseOptionsHelpers",
        "//mediapipe/tasks/ios/core:MPPTaskOptionsProtocol",
        "//mediapipe/tasks/ios/components/processors/utils:MPPClassifierOptionsHelpers",
        "//mediapipe/tasks/cc/text/text_classifier/proto:text_classifier_graph_options_cc_proto",
    ],
)

objc_library(
    name = "MPPTextClassifierResultHelpers",
    srcs = ["sources/MPPTextClassifierResult+Helpers.mm"],
    hdrs = ["sources/MPPTextClassifierResult+Helpers.h"],
    deps = [
        "//mediapipe/tasks/ios/text/text_classifier:MPPTextClassifierResult",
        "//mediapipe/tasks/ios/components/containers/utils:MPPClassificationResultHelpers",
    ],
)
