// Copyright 2019 The MediaPipe Authors.
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

#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/collection_item_id.h"
#include "mediapipe/framework/deps/message_matchers.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

namespace {

::mediapipe::Detection DetectionWithRelativeLocationData(double xmin,
                                                         double ymin,
                                                         double width,
                                                         double height) {
  ::mediapipe::Detection detection;
  ::mediapipe::LocationData* location_data = detection.mutable_location_data();
  location_data->set_format(::mediapipe::LocationData::RELATIVE_BOUNDING_BOX);
  location_data->mutable_relative_bounding_box()->set_xmin(xmin);
  location_data->mutable_relative_bounding_box()->set_ymin(ymin);
  location_data->mutable_relative_bounding_box()->set_width(width);
  location_data->mutable_relative_bounding_box()->set_height(height);
  return detection;
}

}  // namespace

class AssociationDetectionCalculatorTest : public ::testing::Test {
 protected:
  AssociationDetectionCalculatorTest() {
    //  0.4                                         ================
    //                                              |    |    |    |
    //  0.3 =====================                   |  DET2   |    |
    //      |    |    |   DET1  |                   |    |   DET4  |
    //  0.2 |   DET0  |    ===========              ================
    //      |    |    |    |    |    |
    //  0.1 =====|===============    |
    //           |    DET3 |    |    |
    //  0.0      ================    |
    //                     |   DET5  |
    // -0.1                ===========
    //     0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0  1.1  1.2

    // Detection det_0.
    det_0 = DetectionWithRelativeLocationData(/*xmin=*/0.1, /*ymin=*/0.1,
                                              /*width=*/0.2, /*height=*/0.2);
    det_0.set_detection_id(0);

    // Detection det_1.
    det_1 = DetectionWithRelativeLocationData(/*xmin=*/0.3, /*ymin=*/0.1,
                                              /*width=*/0.2, /*height=*/0.2);
    det_1.set_detection_id(1);

    // Detection det_2.
    det_2 = DetectionWithRelativeLocationData(/*xmin=*/0.9, /*ymin=*/0.2,
                                              /*width=*/0.2, /*height=*/0.2);
    det_2.set_detection_id(2);

    // Detection det_3.
    det_3 = DetectionWithRelativeLocationData(/*xmin=*/0.2, /*ymin=*/0.0,
                                              /*width=*/0.3, /*height=*/0.3);
    det_3.set_detection_id(3);

    // Detection det_4.
    det_4 = DetectionWithRelativeLocationData(/*xmin=*/1.0, /*ymin=*/0.2,
                                              /*width=*/0.2, /*height=*/0.2);
    det_4.set_detection_id(4);

    // Detection det_5.
    det_5 = DetectionWithRelativeLocationData(/*xmin=*/0.3, /*ymin=*/-0.1,
                                              /*width=*/0.3, /*height=*/0.3);
    det_5.set_detection_id(5);
  }

  ::mediapipe::Detection det_0, det_1, det_2, det_3, det_4, det_5;
};

TEST_F(AssociationDetectionCalculatorTest, DetectionAssocTest) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
    calculator: "AssociationDetectionCalculator"
    input_stream: "input_vec_0"
    input_stream: "input_vec_1"
    input_stream: "input_vec_2"
    output_stream: "output_vec"
    options {
      [mediapipe.AssociationCalculatorOptions.ext] {
        min_similarity_threshold: 0.1
      }
    }
  )"));

  // Input Stream 0: det_0, det_1, det_2.
  auto input_vec_0 = absl::make_unique<std::vector<::mediapipe::Detection>>();
  input_vec_0->push_back(det_0);
  input_vec_0->push_back(det_1);
  input_vec_0->push_back(det_2);
  runner.MutableInputs()->Index(0).packets.push_back(
      Adopt(input_vec_0.release()).At(Timestamp(1)));

  // Input Stream 1: det_3, det_4.
  auto input_vec_1 = absl::make_unique<std::vector<::mediapipe::Detection>>();
  input_vec_1->push_back(det_3);
  input_vec_1->push_back(det_4);
  runner.MutableInputs()->Index(1).packets.push_back(
      Adopt(input_vec_1.release()).At(Timestamp(1)));

  // Input Stream 2: det_5.
  auto input_vec_2 = absl::make_unique<std::vector<::mediapipe::Detection>>();
  input_vec_2->push_back(det_5);
  runner.MutableInputs()->Index(2).packets.push_back(
      Adopt(input_vec_2.release()).At(Timestamp(1)));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, output.size());
  const auto& assoc_rects =
      output[0].Get<std::vector<::mediapipe::Detection>>();

  // det_3 overlaps with det_0, det_1 and det_5 overlaps with det_3. Since det_5
  // is in the highest priority, we remove other rects. det_4 overlaps with
  // det_2, and det_4 is higher priority, so we keep it. The final output
  // therefore contains 2 elements.
  EXPECT_EQ(2, assoc_rects.size());
  // Outputs are in order of inputs, so det_4 is before det_5 in output vector.

  // det_4 overlaps with det_2, so new id for det_4 is 2.
  EXPECT_TRUE(assoc_rects[0].has_detection_id());
  EXPECT_EQ(2, assoc_rects[0].detection_id());
  det_4.set_detection_id(2);
  EXPECT_THAT(assoc_rects[0], EqualsProto(det_4));

  // det_3 overlaps with det_0, so new id for det_3 is 0.
  // det_3 overlaps with det_1, so new id for det_3 is 1.
  // det_5 overlaps with det_3, so new id for det_5 is 1.
  EXPECT_TRUE(assoc_rects[1].has_detection_id());
  EXPECT_EQ(1, assoc_rects[1].detection_id());
  det_5.set_detection_id(1);
  EXPECT_THAT(assoc_rects[1], EqualsProto(det_5));
}

TEST_F(AssociationDetectionCalculatorTest, DetectionAssocTestWithPrev) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
    calculator: "AssociationDetectionCalculator"
    input_stream: "PREV:input_vec_0"
    input_stream: "input_vec_1"
    output_stream: "output_vec"
    options {
      [mediapipe.AssociationCalculatorOptions.ext] {
        min_similarity_threshold: 0.1
      }
    }
  )"));

  // Input Stream 0: det_3, det_4.
  auto input_vec_0 = absl::make_unique<std::vector<::mediapipe::Detection>>();
  input_vec_0->push_back(det_3);
  input_vec_0->push_back(det_4);
  CollectionItemId prev_input_stream_id =
      runner.MutableInputs()->GetId("PREV", 0);
  runner.MutableInputs()
      ->Get(prev_input_stream_id)
      .packets.push_back(Adopt(input_vec_0.release()).At(Timestamp(1)));

  // Input Stream 1: det_5.
  auto input_vec_1 = absl::make_unique<std::vector<::mediapipe::Detection>>();
  input_vec_1->push_back(det_5);
  CollectionItemId input_stream_id = runner.MutableInputs()->GetId("", 0);
  runner.MutableInputs()
      ->Get(input_stream_id)
      .packets.push_back(Adopt(input_vec_1.release()).At(Timestamp(1)));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, output.size());
  const auto& assoc_rects =
      output[0].Get<std::vector<::mediapipe::Detection>>();

  // det_5 overlaps with det_3 and doesn't overlap with det_4. Since det_4 is
  // in the PREV input stream, it doesn't get copied to the output, so the final
  // output contains 1 element.
  EXPECT_EQ(1, assoc_rects.size());

  // det_5 overlaps with det_3, det_3 is in PREV, so new id for det_5 is 3.
  EXPECT_TRUE(assoc_rects[0].has_detection_id());
  EXPECT_EQ(3, assoc_rects[0].detection_id());
  det_5.set_detection_id(3);
  EXPECT_THAT(assoc_rects[0], EqualsProto(det_5));
}

TEST_F(AssociationDetectionCalculatorTest, DetectionAssocTestReverse) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
    calculator: "AssociationDetectionCalculator"
    input_stream: "input_vec_0"
    input_stream: "input_vec_1"
    input_stream: "input_vec_2"
    output_stream: "output_vec"
    options {
      [mediapipe.AssociationCalculatorOptions.ext] {
        min_similarity_threshold: 0.1
      }
    }
  )"));

  // Input Stream 0: det_5.
  auto input_vec_0 = absl::make_unique<std::vector<::mediapipe::Detection>>();
  input_vec_0->push_back(det_5);
  runner.MutableInputs()->Index(0).packets.push_back(
      Adopt(input_vec_0.release()).At(Timestamp(1)));

  // Input Stream 1: det_3, det_4.
  auto input_vec_1 = absl::make_unique<std::vector<::mediapipe::Detection>>();
  input_vec_1->push_back(det_3);
  input_vec_1->push_back(det_4);
  runner.MutableInputs()->Index(1).packets.push_back(
      Adopt(input_vec_1.release()).At(Timestamp(1)));

  // Input Stream 2: det_0, det_1, det_2.
  auto input_vec_2 = absl::make_unique<std::vector<::mediapipe::Detection>>();
  input_vec_2->push_back(det_0);
  input_vec_2->push_back(det_1);
  input_vec_2->push_back(det_2);
  runner.MutableInputs()->Index(2).packets.push_back(
      Adopt(input_vec_2.release()).At(Timestamp(1)));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, output.size());
  const auto& assoc_rects =
      output[0].Get<std::vector<::mediapipe::Detection>>();

  // det_3 overlaps with det_5, so det_5 is removed. det_0 overlaps with det_3,
  // so det_3 is removed as det_0 is in higher priority for keeping. det_2
  // overlaps with det_4 so det_4 is removed as det_2 is higher priority for
  // keeping. The final output therefore contains 3 elements.
  EXPECT_EQ(3, assoc_rects.size());
  // Outputs are in same order as inputs.

  // det_3 overlaps with det_5, so new id for det_3 is 5.
  // det_0 overlaps with det_3, so new id for det_0 is 5.
  EXPECT_TRUE(assoc_rects[0].has_detection_id());
  EXPECT_EQ(5, assoc_rects[0].detection_id());
  det_0.set_detection_id(5);
  EXPECT_THAT(assoc_rects[0], EqualsProto(det_0));

  // det_1 stays with id 1.
  EXPECT_TRUE(assoc_rects[1].has_detection_id());
  EXPECT_EQ(1, assoc_rects[1].detection_id());
  EXPECT_THAT(assoc_rects[1], EqualsProto(det_1));

  // det_2 overlaps with det_4, so new id for det_2 is 4.
  EXPECT_TRUE(assoc_rects[2].has_detection_id());
  EXPECT_EQ(4, assoc_rects[2].detection_id());
  det_2.set_detection_id(4);
  EXPECT_THAT(assoc_rects[2], EqualsProto(det_2));
}

class AssociationNormRectCalculatorTest : public ::testing::Test {
 protected:
  AssociationNormRectCalculatorTest() {
    //  0.4                                         ================
    //                                              |    |    |    |
    //  0.3 =====================                   |   NR2   |    |
    //      |    |    |   NR1   |                   |    |    NR4  |
    //  0.2 |   NR0   |    ===========              ================
    //      |    |    |    |    |    |
    //  0.1 =====|===============    |
    //           |    NR3  |    |    |
    //  0.0      ================    |
    //                     |   NR5   |
    // -0.1                ===========
    //     0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0  1.1  1.2

    // NormalizedRect nr_0.
    nr_0.set_x_center(0.2);
    nr_0.set_y_center(0.2);
    nr_0.set_width(0.2);
    nr_0.set_height(0.2);

    // NormalizedRect nr_1.
    nr_1.set_x_center(0.4);
    nr_1.set_y_center(0.2);
    nr_1.set_width(0.2);
    nr_1.set_height(0.2);

    // NormalizedRect nr_2.
    nr_2.set_x_center(1.0);
    nr_2.set_y_center(0.3);
    nr_2.set_width(0.2);
    nr_2.set_height(0.2);

    // NormalizedRect nr_3.
    nr_3.set_x_center(0.35);
    nr_3.set_y_center(0.15);
    nr_3.set_width(0.3);
    nr_3.set_height(0.3);

    // NormalizedRect nr_4.
    nr_4.set_x_center(1.1);
    nr_4.set_y_center(0.3);
    nr_4.set_width(0.2);
    nr_4.set_height(0.2);

    // NormalizedRect nr_5.
    nr_5.set_x_center(0.45);
    nr_5.set_y_center(0.05);
    nr_5.set_width(0.3);
    nr_5.set_height(0.3);
  }

  ::mediapipe::NormalizedRect nr_0, nr_1, nr_2, nr_3, nr_4, nr_5;
};

TEST_F(AssociationNormRectCalculatorTest, NormRectAssocTest) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
    calculator: "AssociationNormRectCalculator"
    input_stream: "input_vec_0"
    input_stream: "input_vec_1"
    input_stream: "input_vec_2"
    output_stream: "output_vec"
    options {
      [mediapipe.AssociationCalculatorOptions.ext] {
        min_similarity_threshold: 0.1
      }
    }
  )"));

  // Input Stream 0: nr_0, nr_1, nr_2.
  auto input_vec_0 =
      absl::make_unique<std::vector<::mediapipe::NormalizedRect>>();
  input_vec_0->push_back(nr_0);
  input_vec_0->push_back(nr_1);
  input_vec_0->push_back(nr_2);
  runner.MutableInputs()->Index(0).packets.push_back(
      Adopt(input_vec_0.release()).At(Timestamp(1)));

  // Input Stream 1: nr_3, nr_4.
  auto input_vec_1 =
      absl::make_unique<std::vector<::mediapipe::NormalizedRect>>();
  input_vec_1->push_back(nr_3);
  input_vec_1->push_back(nr_4);
  runner.MutableInputs()->Index(1).packets.push_back(
      Adopt(input_vec_1.release()).At(Timestamp(1)));

  // Input Stream 2: nr_5.
  auto input_vec_2 =
      absl::make_unique<std::vector<::mediapipe::NormalizedRect>>();
  input_vec_2->push_back(nr_5);
  runner.MutableInputs()->Index(2).packets.push_back(
      Adopt(input_vec_2.release()).At(Timestamp(1)));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, output.size());
  const auto& assoc_rects =
      output[0].Get<std::vector<::mediapipe::NormalizedRect>>();

  // nr_3 overlaps with nr_0, nr_1 and nr_5 overlaps with nr_3. Since nr_5 is
  // in the highest priority, we remove other rects.
  // nr_4 overlaps with nr_2, and nr_4 is higher priority, so we keep it.
  // The final output therefore contains 2 elements.
  EXPECT_EQ(2, assoc_rects.size());
  // Outputs are in order of inputs, so nr_4 is before nr_5 in output vector.
  EXPECT_THAT(assoc_rects[0], EqualsProto(nr_4));
  EXPECT_THAT(assoc_rects[1], EqualsProto(nr_5));
}

TEST_F(AssociationNormRectCalculatorTest, NormRectAssocTestReverse) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
    calculator: "AssociationNormRectCalculator"
    input_stream: "input_vec_0"
    input_stream: "input_vec_1"
    input_stream: "input_vec_2"
    output_stream: "output_vec"
    options {
      [mediapipe.AssociationCalculatorOptions.ext] {
        min_similarity_threshold: 0.1
      }
    }
  )"));

  // Input Stream 0: nr_5.
  auto input_vec_0 =
      absl::make_unique<std::vector<::mediapipe::NormalizedRect>>();
  input_vec_0->push_back(nr_5);
  runner.MutableInputs()->Index(0).packets.push_back(
      Adopt(input_vec_0.release()).At(Timestamp(1)));

  // Input Stream 1: nr_3, nr_4.
  auto input_vec_1 =
      absl::make_unique<std::vector<::mediapipe::NormalizedRect>>();
  input_vec_1->push_back(nr_3);
  input_vec_1->push_back(nr_4);
  runner.MutableInputs()->Index(1).packets.push_back(
      Adopt(input_vec_1.release()).At(Timestamp(1)));

  // Input Stream 2: nr_0, nr_1, nr_2.
  auto input_vec_2 =
      absl::make_unique<std::vector<::mediapipe::NormalizedRect>>();
  input_vec_2->push_back(nr_0);
  input_vec_2->push_back(nr_1);
  input_vec_2->push_back(nr_2);
  runner.MutableInputs()->Index(2).packets.push_back(
      Adopt(input_vec_2.release()).At(Timestamp(1)));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, output.size());
  const auto& assoc_rects =
      output[0].Get<std::vector<::mediapipe::NormalizedRect>>();

  // nr_3 overlaps with nr_5, so nr_5 is removed. nr_0 overlaps with nr_3, so
  // nr_3 is removed as nr_0 is in higher priority for keeping. nr_2 overlaps
  // with nr_4 so nr_4 is removed as nr_2 is higher priority for keeping.
  // The final output therefore contains 3 elements.
  EXPECT_EQ(3, assoc_rects.size());
  // Outputs are in same order as inputs.
  EXPECT_THAT(assoc_rects[0], EqualsProto(nr_0));
  EXPECT_THAT(assoc_rects[1], EqualsProto(nr_1));
  EXPECT_THAT(assoc_rects[2], EqualsProto(nr_2));
}

TEST_F(AssociationNormRectCalculatorTest, NormRectAssocSingleInputStream) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
    calculator: "AssociationNormRectCalculator"
    input_stream: "input_vec"
    output_stream: "output_vec"
    options {
      [mediapipe.AssociationCalculatorOptions.ext] {
        min_similarity_threshold: 0.1
      }
    }
  )"));

  // Input Stream : nr_3, nr_5.
  auto input_vec =
      absl::make_unique<std::vector<::mediapipe::NormalizedRect>>();
  input_vec->push_back(nr_3);
  input_vec->push_back(nr_5);
  runner.MutableInputs()->Index(0).packets.push_back(
      Adopt(input_vec.release()).At(Timestamp(1)));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, output.size());
  const auto& assoc_rects =
      output[0].Get<std::vector<::mediapipe::NormalizedRect>>();

  // nr_5 overlaps with nr_3. Since nr_5 is after nr_3 in the same input stream
  // we remove nr_3 and keep nr_5.
  // The final output therefore contains 1 elements.
  EXPECT_EQ(1, assoc_rects.size());
  EXPECT_THAT(assoc_rects[0], EqualsProto(nr_5));
}

}  // namespace mediapipe
