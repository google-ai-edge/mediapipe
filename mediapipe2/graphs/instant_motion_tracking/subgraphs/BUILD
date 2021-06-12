# Copyright 2020 Google LLC
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

load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_simple_subgraph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_simple_subgraph(
    name = "region_tracking",
    graph = "region_tracking.pbtxt",
    register_as = "RegionTrackingSubgraph",
    deps = [
        "//mediapipe/graphs/instant_motion_tracking/calculators:tracked_anchor_manager_calculator",
        "//mediapipe/graphs/tracking/subgraphs:box_tracking_gpu",
    ],
)
