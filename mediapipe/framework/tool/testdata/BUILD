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
#

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//mediapipe:__subpackages__"])

load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_simple_subgraph",
)

filegroup(
    name = "test_graph",
    srcs = ["test.pbtxt"],
)

exports_files([
    "test.pbtxt",
    "dub_quad_test_subgraph.pbtxt",
    "nested_test_subgraph.pbtxt",
])

mediapipe_simple_subgraph(
    name = "dub_quad_test_subgraph",
    testonly = 1,
    graph = "dub_quad_test_subgraph.pbtxt",
    register_as = "DubQuadTestSubgraph",
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:test_calculators",
    ],
)

mediapipe_simple_subgraph(
    name = "nested_test_subgraph",
    testonly = 1,
    graph = "nested_test_subgraph.pbtxt",
    register_as = "NestedTestSubgraph",
    visibility = ["//visibility:public"],
    deps = [
        ":dub_quad_test_subgraph",
        "//mediapipe/framework:test_calculators",
    ],
)
