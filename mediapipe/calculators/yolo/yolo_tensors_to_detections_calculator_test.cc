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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace tasks {
namespace components {
namespace processors {
namespace {

using ::mediapipe::Tensor;

// Build a CalculatorRunner pre-configured with the given options pbtxt snippet.
std::unique_ptr<CalculatorRunner> MakeRunner(
    const std::string& options_pbtxt) {
  auto node = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
      absl::StrCat(R"pb(
        calculator: "YoloTensorsToDetectionsCalculator"
        input_stream:  "TENSORS:tensors"
        output_stream: "DETECTIONS:detections"
        options {
          [mediapipe.tasks.components.processors.proto
            .YoloTensorsToDetectionsCalculatorOptions.ext] { )pb",
        options_pbtxt, R"pb( } } )pb"));
  return std::make_unique<CalculatorRunner>(node);
}

// Helper: create a float32 Tensor with the given shape and flat data.
Tensor MakeFloatTensor(std::vector<int> dims, std::vector<float> flat_data) {
  Tensor t(Tensor::ElementType::kFloat32, Tensor::Shape(dims));
  auto view = t.GetCpuWriteView();
  std::copy(flat_data.begin(), flat_data.end(), view.buffer<float>());
  return t;
}

// Run calculator with one input tensor, return output detections.
std::vector<Detection> Run(CalculatorRunner& runner, Tensor tensor) {
  std::vector<Tensor> tensors;
  tensors.push_back(std::move(tensor));
  runner.MutableInputs()
      ->Tag("TENSORS")
      .packets.push_back(
          MakePacket<std::vector<Tensor>>(std::move(tensors)).At(Timestamp(0)));
  MP_EXPECT_OK(runner.Run());
  const auto& out = runner.Outputs().Tag("DETECTIONS").packets;
  if (out.empty()) return {};
  return out[0].Get<std::vector<Detection>>();
}

// ── Family A: ULTRALYTICS FEATURES_FIRST ──────────────────────────────────
// Shape [7, 5]: 4 box coords + 3 class scores, 5 anchor candidates.
// Layout FEATURES_FIRST: data[feature * num_boxes + box].
// Plant anchor 0 with cx=320, cy=240, w=100, h=80, class 1 score=0.9.
TEST(YoloTensorsToDetectionsTest, FamilyA_FeaturesFirst_OneDetection) {
  const int num_features = 7, num_boxes = 5;
  std::vector<float> data(num_features * num_boxes, 0.0f);
  data[0 * num_boxes + 0] = 320.0f;  // cx
  data[1 * num_boxes + 0] = 240.0f;  // cy
  data[2 * num_boxes + 0] = 100.0f;  // w
  data[3 * num_boxes + 0] = 80.0f;   // h
  data[5 * num_boxes + 0] = 0.9f;    // class 1 score (feature index 4+1)

  auto runner = MakeRunner(
      "decode_mode: ULTRALYTICS_DETECTION_HEAD "
      "tensor_layout: FEATURES_FIRST "
      "input_width: 640 input_height: 640 "
      "min_score_threshold: 0.5");
  auto dets = Run(*runner, MakeFloatTensor({num_features, num_boxes}, data));

  ASSERT_EQ(dets.size(), 1u);
  EXPECT_EQ(dets[0].label_id(0), 1);
  EXPECT_NEAR(dets[0].score(0), 0.9f, 1e-5f);

  // cx=320, cy=240, w=100, h=80 on 640×640 → xmin=(320-50)/640, ymin=(240-40)/640
  const auto& bb = dets[0].location_data().relative_bounding_box();
  EXPECT_NEAR(bb.xmin(),   270.0f / 640.0f, 1e-4f);
  EXPECT_NEAR(bb.ymin(),   200.0f / 640.0f, 1e-4f);
  EXPECT_NEAR(bb.width(),  100.0f / 640.0f, 1e-4f);
  EXPECT_NEAR(bb.height(),  80.0f / 640.0f, 1e-4f);
}

// ── Family A: ULTRALYTICS BOXES_FIRST ────────────────────────────────────
TEST(YoloTensorsToDetectionsTest, FamilyA_BoxesFirst_OneDetection) {
  const int num_boxes = 5, num_features = 7;
  std::vector<float> data(num_boxes * num_features, 0.0f);
  // BOXES_FIRST: data[box * num_features + feature]
  data[0 * num_features + 0] = 320.0f;  // cx
  data[0 * num_features + 1] = 240.0f;  // cy
  data[0 * num_features + 2] = 100.0f;  // w
  data[0 * num_features + 3] = 80.0f;   // h
  data[0 * num_features + 5] = 0.9f;    // class 1

  auto runner = MakeRunner(
      "decode_mode: ULTRALYTICS_DETECTION_HEAD "
      "tensor_layout: BOXES_FIRST "
      "input_width: 640 input_height: 640 "
      "min_score_threshold: 0.5");
  auto dets = Run(*runner, MakeFloatTensor({num_boxes, num_features}, data));

  ASSERT_EQ(dets.size(), 1u);
  EXPECT_EQ(dets[0].label_id(0), 1);
  EXPECT_NEAR(dets[0].score(0), 0.9f, 1e-5f);
}

// ── Family B: END_TO_END BOXES_FIRST ─────────────────────────────────────
// Shape [3, 6]: 3 detections × [x1,y1,x2,y2,score,class_id].
// threshold=0.5: box 0 (score=0.85) and box 2 (score=0.75) pass; box 1 (0.10) filtered.
TEST(YoloTensorsToDetectionsTest, FamilyB_EndToEnd_TwoDetections) {
  // clang-format off
  std::vector<float> data = {
      10.0f, 20.0f, 60.0f,  80.0f, 0.85f, 2.0f,  // box 0
       0.0f,  0.0f,  0.0f,   0.0f, 0.10f, 0.0f,  // box 1 — below threshold
      50.0f, 50.0f, 90.0f,  90.0f, 0.75f, 5.0f,  // box 2
  };
  // clang-format on

  auto runner = MakeRunner(
      "decode_mode: END_TO_END "
      "tensor_layout: BOXES_FIRST "
      "input_width: 640 input_height: 640 "
      "min_score_threshold: 0.5");
  auto dets = Run(*runner, MakeFloatTensor({3, 6}, data));

  ASSERT_EQ(dets.size(), 2u);
  EXPECT_EQ(dets[0].label_id(0), 2);
  EXPECT_NEAR(dets[0].score(0), 0.85f, 1e-5f);
  EXPECT_EQ(dets[1].label_id(0), 5);
  EXPECT_NEAR(dets[1].score(0), 0.75f, 1e-5f);

  // Relative bbox for box 0: x1/640, y1/640, (x2-x1)/640, (y2-y1)/640
  const auto& bb = dets[0].location_data().relative_bounding_box();
  EXPECT_NEAR(bb.xmin(),   10.0f / 640.0f, 1e-4f);
  EXPECT_NEAR(bb.ymin(),   20.0f / 640.0f, 1e-4f);
  EXPECT_NEAR(bb.width(),  50.0f / 640.0f, 1e-4f);
  EXPECT_NEAR(bb.height(), 60.0f / 640.0f, 1e-4f);

  const auto& bb2 = dets[1].location_data().relative_bounding_box();
  EXPECT_NEAR(bb2.xmin(),   50.0f / 640.0f, 1e-4f);
  EXPECT_NEAR(bb2.ymin(),   50.0f / 640.0f, 1e-4f);
  EXPECT_NEAR(bb2.width(),  40.0f / 640.0f, 1e-4f);
  EXPECT_NEAR(bb2.height(), 40.0f / 640.0f, 1e-4f);
}

// ── AUTO: shape [3,6] → END_TO_END ────────────────────────────────────────
TEST(YoloTensorsToDetectionsTest, Auto_DetectsEndToEndFromDimEqualsSix) {
  std::vector<float> data(3 * 6, 0.0f);
  data[4] = 0.9f;  // score of box 0
  data[5] = 1.0f;  // class_id of box 0

  auto runner = MakeRunner(
      "decode_mode: DECODE_MODE_AUTO "
      "input_width: 640 input_height: 640 "
      "min_score_threshold: 0.5");
  auto dets = Run(*runner, MakeFloatTensor({3, 6}, data));

  ASSERT_EQ(dets.size(), 1u);
  EXPECT_EQ(dets[0].label_id(0), 1);
}

// ── AUTO: shape [7,5] → ULTRALYTICS ──────────────────────────────────────
TEST(YoloTensorsToDetectionsTest, Auto_DetectsUltralyticsFromShape) {
  const int num_boxes = 7, num_features = 5;
  std::vector<float> data(num_boxes * num_features, 0.0f);
  // BOXES_FIRST auto-detected (7>5, 5>4, 5<=512): box 0, class 0 (feature index 4)
  data[0 * num_features + 4] = 0.9f;  // box 0, class 0 score

  auto runner = MakeRunner(
      "decode_mode: DECODE_MODE_AUTO "
      "input_width: 640 input_height: 640 "
      "min_score_threshold: 0.5");
  auto dets = Run(*runner, MakeFloatTensor({num_boxes, num_features}, data));

  EXPECT_EQ(dets.size(), 1u);
  EXPECT_EQ(dets[0].label_id(0), 0);
}

// ── Ambiguous [6,6] with explicit END_TO_END succeeds ────────────────────
TEST(YoloTensorsToDetectionsTest, Explicit_EndToEnd_AmbiguousShape) {
  std::vector<float> data(6 * 6, 0.0f);
  data[4] = 0.9f;  // score for box 0
  data[5] = 0.0f;  // class_id 0

  auto runner = MakeRunner(
      "decode_mode: END_TO_END "
      "tensor_layout: BOXES_FIRST "
      "input_width: 640 input_height: 640 "
      "min_score_threshold: 0.5");
  auto dets = Run(*runner, MakeFloatTensor({6, 6}, data));

  EXPECT_EQ(dets.size(), 1u);
}

// ── Below-threshold scores produce no detections ─────────────────────────
TEST(YoloTensorsToDetectionsTest, AllBelowThreshold_NoDetections) {
  std::vector<float> data(3 * 6, 0.0f);  // all scores = 0

  auto runner = MakeRunner(
      "decode_mode: END_TO_END "
      "input_width: 640 input_height: 640 "
      "min_score_threshold: 0.5");
  auto dets = Run(*runner, MakeFloatTensor({3, 6}, data));

  EXPECT_EQ(dets.size(), 0u);
}

// ── INT8 quantized END_TO_END with quantization_scale_override ────────────
// INT8 tensor: values are int8, dequantized via scale*(val - zero_point).
// Box 0: raw coords [10,20,60,80] (pixel), score raw=90, class=2.
// With scale=1.0, zero_point=0 → dequantized values equal raw values.
// Score dequantized: 90.0 → above threshold 0.5. Passes.
TEST(YoloTensorsToDetectionsTest, Int8_EndToEnd_QuantizationScaleOverride) {
  // shape [2, 6]: 2 boxes
  std::vector<int8_t> raw = {
      10, 20, 60, 80, 90, 2,   // box 0: coords + score=90*scale + class=2
       0,  0,  0,  0,  0, 0,   // box 1: all zero
  };
  Tensor t(Tensor::ElementType::kInt8, Tensor::Shape({2, 6}));
  {
    auto view = t.GetCpuWriteView();
    std::copy(raw.begin(), raw.end(), view.buffer<int8_t>());
  }

  auto runner = MakeRunner(
      "decode_mode: END_TO_END "
      "tensor_layout: BOXES_FIRST "
      "input_width: 640 input_height: 640 "
      "min_score_threshold: 0.5 "
      "quantization_scale_override: 1.0 "
      "quantization_zero_point_override: 0");

  std::vector<Tensor> tensors;
  tensors.push_back(std::move(t));
  runner->MutableInputs()->Tag("TENSORS").packets.push_back(
      MakePacket<std::vector<Tensor>>(std::move(tensors)).At(Timestamp(0)));
  MP_ASSERT_OK(runner->Run());

  const auto& out = runner->Outputs().Tag("DETECTIONS").packets;
  ASSERT_EQ(out.size(), 1u);
  const auto& dets = out[0].Get<std::vector<Detection>>();
  ASSERT_EQ(dets.size(), 1u);
  EXPECT_EQ(dets[0].label_id(0), 2);
  EXPECT_NEAR(dets[0].score(0), 90.0f, 0.1f);
}

// ── Mismatched quantization override → kInvalidArgument ──────────────────
// Providing scale without zero_point (or vice versa) must fail in Open().
TEST(YoloTensorsToDetectionsTest, MismatchedQuantizationOverride_ReturnsError) {
  auto node = ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "YoloTensorsToDetectionsCalculator"
    input_stream:  "TENSORS:tensors"
    output_stream: "DETECTIONS:detections"
    options {
      [mediapipe.tasks.components.processors.proto
        .YoloTensorsToDetectionsCalculatorOptions.ext] {
        decode_mode: END_TO_END
        input_width: 640
        input_height: 640
        quantization_scale_override: 1.0
        # intentionally omitting quantization_zero_point_override
      }
    }
  )pb");
  CalculatorRunner runner(node);
  // Run with a dummy tensor — Open() should fail before Process() is reached.
  std::vector<Tensor> tensors;
  tensors.push_back(
      MakeFloatTensor({3, 6}, std::vector<float>(3 * 6, 0.0f)));
  runner.MutableInputs()->Tag("TENSORS").packets.push_back(
      MakePacket<std::vector<Tensor>>(std::move(tensors)).At(Timestamp(0)));
  EXPECT_THAT(runner.Run(),
              mediapipe::StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace processors
}  // namespace components
}  // namespace tasks
}  // namespace mediapipe
