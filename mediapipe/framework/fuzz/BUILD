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

load("@rules_fuzzing//fuzzing:cc_defs.bzl", "cc_fuzz_test")

package(default_visibility = ["//visibility:private"])

cc_fuzz_test(
    name = "calculator_graph_config_fuzz",
    srcs = ["calculator_graph_config_fuzz.cc"],
    corpus = glob(["corpus/**"], allow_empty = True),
    deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_graph",
    ],
)
