// Copyright 2026 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include "mediapipe/calculators/tflite/tflite_model_metadata.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

Detection MakeDet(float xmin, float ymin, float w, float h, float score) {
  Detection d;
  d.add_score(score);
  d.add_label_id(0);
  auto* bb = d.mutable_location_data()->mutable_relative_bounding_box();
  bb->set_xmin(xmin);
  bb->set_ymin(ymin);
  bb->set_width(w);
  bb->set_height(h);
  return d;
}

// Build a CalculatorGraph that runs AutoNmsCalculator.
// `node_pbtxt_extra` is appended inside the node stanza (e.g. side-packet
// input stream declaration).
// `graph_extra` is appended at the top level (e.g. input_side_packet decl).
CalculatorGraph BuildGraph(const std::string& graph_extra,
                           const std::string& node_extra,
                           const std::string& options_pbtxt) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(
      absl::StrCat(R"pb(
        input_stream: "detections"
        output_stream: "out"
      )pb",
      graph_extra, R"pb(
        node {
          calculator: "AutoNmsCalculator"
          input_stream:  "DETECTIONS:detections"
          output_stream: "DETECTIONS:out"
      )pb",
      node_extra, R"pb(
          options {
            [mediapipe.AutoNmsCalculatorOptions.ext] { )pb",
      options_pbtxt, R"pb( }
          }
        }
      )pb"));
  return CalculatorGraph(config);
}

// Run the graph with the given detections and optional side packets.
// Returns the output detections vector (from the single output packet).
std::vector<Detection> RunGraph(
    CalculatorGraph& graph,
    const std::map<std::string, Packet>& side_packets,
    const std::vector<Detection>& input_dets) {
  std::vector<Detection> result;

  // Observe output
  MP_EXPECT_OK(graph.ObserveOutputStream(
      "out", [&result](const Packet& p) -> absl::Status {
        result = p.Get<std::vector<Detection>>();
        return absl::OkStatus();
      }));

  MP_EXPECT_OK(graph.StartRun(side_packets));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "detections",
      MakePacket<std::vector<Detection>>(input_dets).At(Timestamp(0))));
  MP_EXPECT_OK(graph.CloseInputStream("detections"));
  MP_EXPECT_OK(graph.WaitUntilDone());
  return result;
}

// ---------------------------------------------------------------------------
// Test 1: end-to-end metadata (output shape contains dim==6) → skip NMS
// ---------------------------------------------------------------------------
TEST(AutoNmsCalculatorTest, EndToEndMetadata_SkipsNms) {
  // Two fully-overlapping boxes (IoU == 1.0). If NMS ran, only the
  // higher-score box would survive. If skipped, both must pass through.
  auto graph = BuildGraph(
      /*graph_extra=*/"input_side_packet: \"metadata\"",
      /*node_extra=*/"input_side_packet: \"MODEL_METADATA:metadata\"",
      /*options_pbtxt=*/"iou_threshold: 0.45");

  TfLiteModelMetadata meta;
  auto* out = meta.add_outputs();
  out->add_shape(300);
  out->add_shape(6);  // dim==6 → end-to-end model

  std::vector<Detection> dets = {
      MakeDet(0.1f, 0.1f, 0.5f, 0.5f, 0.9f),
      MakeDet(0.1f, 0.1f, 0.5f, 0.5f, 0.7f),
  };

  std::map<std::string, Packet> side_packets = {
      {"metadata", MakePacket<TfLiteModelMetadata>(meta)}};

  auto result = RunGraph(graph, side_packets, dets);
  EXPECT_EQ(result.size(), 2u) << "NMS should be skipped; both boxes pass.";
}

// ---------------------------------------------------------------------------
// Test 2: Ultralytics-style metadata (shape [1,84,8400], no dim==6) → run NMS
// ---------------------------------------------------------------------------
TEST(AutoNmsCalculatorTest, UltralyticsMetadata_AppliesNms) {
  auto graph = BuildGraph(
      /*graph_extra=*/"input_side_packet: \"metadata\"",
      /*node_extra=*/"input_side_packet: \"MODEL_METADATA:metadata\"",
      /*options_pbtxt=*/"iou_threshold: 0.45");

  TfLiteModelMetadata meta;
  auto* out = meta.add_outputs();
  out->add_shape(1);
  out->add_shape(84);
  out->add_shape(8400);  // no dim==6 → NMS should run

  std::vector<Detection> dets = {
      MakeDet(0.1f, 0.1f, 0.5f, 0.5f, 0.9f),
      MakeDet(0.1f, 0.1f, 0.5f, 0.5f, 0.7f),
  };

  std::map<std::string, Packet> side_packets = {
      {"metadata", MakePacket<TfLiteModelMetadata>(meta)}};

  auto result = RunGraph(graph, side_packets, dets);
  EXPECT_EQ(result.size(), 1u) << "NMS should suppress the lower-score box.";
  EXPECT_NEAR(result[0].score(0), 0.9f, 1e-5f);
}

// ---------------------------------------------------------------------------
// Test 3: no MODEL_METADATA connected → NMS runs by default
// ---------------------------------------------------------------------------
TEST(AutoNmsCalculatorTest, NoMetadata_RunsNms) {
  // No input_side_packet declaration, no node-level connection.
  auto graph = BuildGraph(
      /*graph_extra=*/"",
      /*node_extra=*/"",
      /*options_pbtxt=*/"iou_threshold: 0.45");

  std::vector<Detection> dets = {
      MakeDet(0.2f, 0.2f, 0.4f, 0.4f, 0.8f),
      MakeDet(0.2f, 0.2f, 0.4f, 0.4f, 0.6f),
  };

  // No side packets
  auto result = RunGraph(graph, {}, dets);
  EXPECT_EQ(result.size(), 1u) << "NMS should suppress the lower-score box.";
  EXPECT_NEAR(result[0].score(0), 0.8f, 1e-5f);
}

// ---------------------------------------------------------------------------
// Test 4: explicit SKIP_NMS in options → always skips, regardless of metadata
// ---------------------------------------------------------------------------
TEST(AutoNmsCalculatorTest, ExplicitSkipNms_IgnoresMetadata) {
  // Metadata does NOT indicate end-to-end (no dim==6), but options force skip.
  auto graph = BuildGraph(
      /*graph_extra=*/"input_side_packet: \"metadata\"",
      /*node_extra=*/"input_side_packet: \"MODEL_METADATA:metadata\"",
      /*options_pbtxt=*/"iou_threshold: 0.45  postprocess_mode: SKIP_NMS");

  TfLiteModelMetadata meta;
  auto* out = meta.add_outputs();
  out->add_shape(1);
  out->add_shape(84);
  out->add_shape(8400);

  std::vector<Detection> dets = {
      MakeDet(0.0f, 0.0f, 0.6f, 0.6f, 0.9f),
      MakeDet(0.0f, 0.0f, 0.6f, 0.6f, 0.5f),
  };

  std::map<std::string, Packet> side_packets = {
      {"metadata", MakePacket<TfLiteModelMetadata>(meta)}};

  auto result = RunGraph(graph, side_packets, dets);
  EXPECT_EQ(result.size(), 2u) << "SKIP_NMS option should bypass NMS.";
}

// ---------------------------------------------------------------------------
// Test 5: empty input detections → empty output, no crash
// ---------------------------------------------------------------------------
TEST(AutoNmsCalculatorTest, EmptyDetections_PassThrough) {
  auto graph = BuildGraph(
      /*graph_extra=*/"",
      /*node_extra=*/"",
      /*options_pbtxt=*/"iou_threshold: 0.45");

  auto result = RunGraph(graph, {}, {});
  // Empty input → calculator returns early (no output packet emitted),
  // so result vector stays empty.
  EXPECT_TRUE(result.empty());
}

}  // namespace
}  // namespace mediapipe
