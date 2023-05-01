/* Copyright 2022 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>
#include <vector>

#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"

namespace mediapipe {
namespace {

using ::mediapipe::NormalizedRect;
using ::testing::ElementsAre;
using ::testing::EqualsProto;

class HandAssociationCalculatorTest : public testing::Test {
 protected:
  HandAssociationCalculatorTest() {
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
    nr_0_.set_x_center(0.2);
    nr_0_.set_y_center(0.2);
    nr_0_.set_width(0.2);
    nr_0_.set_height(0.2);

    // NormalizedRect nr_1.
    nr_1_.set_x_center(0.4);
    nr_1_.set_y_center(0.2);
    nr_1_.set_width(0.2);
    nr_1_.set_height(0.2);

    // NormalizedRect nr_2.
    nr_2_.set_x_center(1.0);
    nr_2_.set_y_center(0.3);
    nr_2_.set_width(0.2);
    nr_2_.set_height(0.2);

    // NormalizedRect nr_3.
    nr_3_.set_x_center(0.35);
    nr_3_.set_y_center(0.15);
    nr_3_.set_width(0.3);
    nr_3_.set_height(0.3);

    // NormalizedRect nr_4.
    nr_4_.set_x_center(1.1);
    nr_4_.set_y_center(0.3);
    nr_4_.set_width(0.2);
    nr_4_.set_height(0.2);

    // NormalizedRect nr_5.
    nr_5_.set_x_center(0.5);
    nr_5_.set_y_center(0.05);
    nr_5_.set_width(0.3);
    nr_5_.set_height(0.3);
  }

  NormalizedRect nr_0_, nr_1_, nr_2_, nr_3_, nr_4_, nr_5_;
};

TEST_F(HandAssociationCalculatorTest, NormRectAssocTest) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "HandAssociationCalculator"
    input_stream: "BASE_RECTS:input_vec_0"
    input_stream: "RECTS:0:input_vec_1"
    input_stream: "RECTS:1:input_vec_2"
    output_stream: "output_vec"
    options {
      [mediapipe.HandAssociationCalculatorOptions.ext] {
        min_similarity_threshold: 0.1
      }
    }
  )pb"));

  // Input Stream 0: nr_0, nr_1, nr_2.
  auto input_vec_0 = std::make_unique<std::vector<NormalizedRect>>();
  input_vec_0->push_back(nr_0_);
  input_vec_0->push_back(nr_1_);
  input_vec_0->push_back(nr_2_);
  runner.MutableInputs()
      ->Tag("BASE_RECTS")
      .packets.push_back(Adopt(input_vec_0.release()).At(Timestamp(1)));

  // Input Stream 1: nr_3, nr_4.
  auto input_vec_1 = std::make_unique<std::vector<NormalizedRect>>();
  input_vec_1->push_back(nr_3_);
  input_vec_1->push_back(nr_4_);
  auto index_id = runner.MutableInputs()->GetId("RECTS", 0);
  runner.MutableInputs()->Get(index_id).packets.push_back(
      Adopt(input_vec_1.release()).At(Timestamp(1)));

  // Input Stream 2: nr_5.
  auto input_vec_2 = std::make_unique<std::vector<NormalizedRect>>();
  input_vec_2->push_back(nr_5_);
  index_id = runner.MutableInputs()->GetId("RECTS", 1);
  runner.MutableInputs()->Get(index_id).packets.push_back(
      Adopt(input_vec_2.release()).At(Timestamp(1)));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, output.size());
  auto assoc_rects = output[0].Get<std::vector<NormalizedRect>>();

  // Rectangles are added in the following sequence:
  // nr_0 is added 1st.
  // nr_1 is added because it does not overlap with nr_0.
  // nr_2 is added because it does not overlap with nr_0 or nr_1.
  // nr_3 is NOT added because it overlaps with nr_0.
  // nr_4 is NOT added because it overlaps with nr_2.
  // nr_5 is NOT added because it overlaps with nr_1.
  EXPECT_EQ(3, assoc_rects.size());

  // Check that IDs are filled in and contents match.
  nr_0_.set_rect_id(1);
  nr_1_.set_rect_id(2);
  nr_2_.set_rect_id(3);
  EXPECT_THAT(assoc_rects, ElementsAre(EqualsProto(nr_0_), EqualsProto(nr_1_),
                                       EqualsProto(nr_2_)));
}

TEST_F(HandAssociationCalculatorTest, NormRectAssocTestWithTrackedHands) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "HandAssociationCalculator"
    input_stream: "BASE_RECTS:input_vec_0"
    input_stream: "RECTS:0:input_vec_1"
    output_stream: "output_vec"
    options {
      [mediapipe.HandAssociationCalculatorOptions.ext] {
        min_similarity_threshold: 0.1
      }
    }
  )pb"));

  // Input Stream 0: nr_0, nr_1.  Tracked hands.
  auto input_vec_0 = std::make_unique<std::vector<NormalizedRect>>();
  // Setting ID to a negative number for test only, since newly generated
  // ID by HandAssociationCalculator are positive numbers.
  nr_0_.set_rect_id(-2);
  input_vec_0->push_back(nr_0_);
  nr_1_.set_rect_id(-1);
  input_vec_0->push_back(nr_1_);
  runner.MutableInputs()
      ->Tag("BASE_RECTS")
      .packets.push_back(Adopt(input_vec_0.release()).At(Timestamp(1)));

  // Input Stream 1: nr_2, nr_3. Newly detected palms.
  auto input_vec_1 = std::make_unique<std::vector<NormalizedRect>>();
  input_vec_1->push_back(nr_2_);
  input_vec_1->push_back(nr_3_);
  runner.MutableInputs()->Tag("RECTS").packets.push_back(
      Adopt(input_vec_1.release()).At(Timestamp(1)));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, output.size());
  auto assoc_rects = output[0].Get<std::vector<NormalizedRect>>();

  // Rectangles are added in the following sequence:
  // nr_0 is added 1st.
  // nr_1 is added because it does not overlap with nr_0.
  // nr_2 is added because it does not overlap with nr_0 or nr_1.
  // nr_3 is NOT added because it overlaps with nr_0.
  EXPECT_EQ(3, assoc_rects.size());

  // Check that IDs are filled in and contents match.
  nr_2_.set_rect_id(1);
  EXPECT_THAT(assoc_rects, ElementsAre(EqualsProto(nr_0_), EqualsProto(nr_1_),
                                       EqualsProto(nr_2_)));
}

TEST_F(HandAssociationCalculatorTest, NormRectAssocTestReverse) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "HandAssociationCalculator"
    input_stream: "BASE_RECTS:input_vec_0"
    input_stream: "RECTS:0:input_vec_1"
    input_stream: "RECTS:1:input_vec_2"
    output_stream: "output_vec"
    options {
      [mediapipe.HandAssociationCalculatorOptions.ext] {
        min_similarity_threshold: 0.1
      }
    }
  )pb"));

  // Input Stream 0: nr_5.
  auto input_vec_0 = std::make_unique<std::vector<NormalizedRect>>();
  input_vec_0->push_back(nr_5_);
  runner.MutableInputs()
      ->Tag("BASE_RECTS")
      .packets.push_back(Adopt(input_vec_0.release()).At(Timestamp(1)));

  // Input Stream 1: nr_4, nr_3
  auto input_vec_1 = std::make_unique<std::vector<NormalizedRect>>();
  input_vec_1->push_back(nr_4_);
  input_vec_1->push_back(nr_3_);
  auto index_id = runner.MutableInputs()->GetId("RECTS", 0);
  runner.MutableInputs()->Get(index_id).packets.push_back(
      Adopt(input_vec_1.release()).At(Timestamp(1)));

  // Input Stream 2: nr_2, nr_1, nr_0.
  auto input_vec_2 = std::make_unique<std::vector<NormalizedRect>>();
  input_vec_2->push_back(nr_2_);
  input_vec_2->push_back(nr_1_);
  input_vec_2->push_back(nr_0_);
  index_id = runner.MutableInputs()->GetId("RECTS", 1);
  runner.MutableInputs()->Get(index_id).packets.push_back(
      Adopt(input_vec_2.release()).At(Timestamp(1)));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, output.size());
  auto assoc_rects = output[0].Get<std::vector<NormalizedRect>>();

  // Rectangles are added in the following sequence:
  // nr_5 is added 1st.
  // nr_4 is added because it does not overlap with nr_5.
  // nr_3 is NOT added because it overlaps with nr_5.
  // nr_2 is NOT added because it overlaps with nr_4.
  // nr_1 is NOT added because it overlaps with nr_5.
  // nr_0 is added because it does not overlap with nr_5 or nr_4.
  EXPECT_EQ(3, assoc_rects.size());

  // Outputs are in same order as inputs, and IDs are filled in.
  nr_5_.set_rect_id(1);
  nr_4_.set_rect_id(2);
  nr_0_.set_rect_id(3);
  EXPECT_THAT(assoc_rects, ElementsAre(EqualsProto(nr_5_), EqualsProto(nr_4_),
                                       EqualsProto(nr_0_)));
}

TEST_F(HandAssociationCalculatorTest, NormRectAssocTestReservesBaseRects) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "HandAssociationCalculator"
    input_stream: "BASE_RECTS:input_vec_0"
    input_stream: "RECTS:0:input_vec_1"
    input_stream: "RECTS:1:input_vec_2"
    output_stream: "output_vec"
    options {
      [mediapipe.HandAssociationCalculatorOptions.ext] {
        min_similarity_threshold: 0.1
      }
    }
  )pb"));

  // Input Stream 0: nr_5, nr_3, nr_1.
  auto input_vec_0 = std::make_unique<std::vector<NormalizedRect>>();
  input_vec_0->push_back(nr_5_);
  input_vec_0->push_back(nr_3_);
  input_vec_0->push_back(nr_1_);
  runner.MutableInputs()
      ->Tag("BASE_RECTS")
      .packets.push_back(Adopt(input_vec_0.release()).At(Timestamp(1)));

  // Input Stream 1: nr_4.
  auto input_vec_1 = std::make_unique<std::vector<NormalizedRect>>();
  input_vec_1->push_back(nr_4_);
  auto index_id = runner.MutableInputs()->GetId("RECTS", 0);
  runner.MutableInputs()->Get(index_id).packets.push_back(
      Adopt(input_vec_1.release()).At(Timestamp(1)));

  // Input Stream 2: nr_2, nr_0.
  auto input_vec_2 = std::make_unique<std::vector<NormalizedRect>>();
  input_vec_2->push_back(nr_2_);
  input_vec_2->push_back(nr_0_);
  index_id = runner.MutableInputs()->GetId("RECTS", 1);
  runner.MutableInputs()->Get(index_id).packets.push_back(
      Adopt(input_vec_2.release()).At(Timestamp(1)));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, output.size());
  auto assoc_rects = output[0].Get<std::vector<NormalizedRect>>();

  // Rectangles are added in the following sequence:
  // nr_5 is added because it is in BASE_RECTS input stream.
  // nr_3 is added because it is in BASE_RECTS input stream.
  // nr_1 is added because it is in BASE_RECTS input stream.
  // nr_4 is added because it does not overlap with nr_5.
  // nr_2 is NOT added because it overlaps with nr_4.
  // nr_0 is NOT added because it overlaps with nr_3.
  EXPECT_EQ(4, assoc_rects.size());

  // Outputs are in same order as inputs, and IDs are filled in.
  nr_5_.set_rect_id(1);
  nr_3_.set_rect_id(2);
  nr_1_.set_rect_id(3);
  nr_4_.set_rect_id(4);
  EXPECT_THAT(assoc_rects, ElementsAre(EqualsProto(nr_5_), EqualsProto(nr_3_),
                                       EqualsProto(nr_1_), EqualsProto(nr_4_)));
}

TEST_F(HandAssociationCalculatorTest, NormRectAssocSingleInputStream) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "HandAssociationCalculator"
    input_stream: "BASE_RECTS:input_vec"
    output_stream: "output_vec"
    options {
      [mediapipe.HandAssociationCalculatorOptions.ext] {
        min_similarity_threshold: 0.1
      }
    }
  )pb"));

  // Just one input stream : nr_3, nr_5.
  auto input_vec = std::make_unique<std::vector<NormalizedRect>>();
  input_vec->push_back(nr_3_);
  input_vec->push_back(nr_5_);
  runner.MutableInputs()
      ->Tag("BASE_RECTS")
      .packets.push_back(Adopt(input_vec.release()).At(Timestamp(1)));

  MP_ASSERT_OK(runner.Run()) << "Calculator execution failed.";
  const std::vector<Packet>& output = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, output.size());
  auto assoc_rects = output[0].Get<std::vector<NormalizedRect>>();

  // Rectangles are added in the following sequence:
  // nr_3 is added 1st.
  // nr_5 is added 2nd. The calculator assumes it does not overlap with nr_3.
  EXPECT_EQ(2, assoc_rects.size());

  nr_3_.set_rect_id(1);
  nr_5_.set_rect_id(2);
  EXPECT_THAT(assoc_rects, ElementsAre(EqualsProto(nr_3_), EqualsProto(nr_5_)));
}

}  // namespace
}  // namespace mediapipe
