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
#include "mediapipe/tasks/cc/vision/hand_landmarker/calculators/hand_association_calculator.h"

#include <utility>
#include <vector>

#include "mediapipe/framework/api3/function_runner.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/packet.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace {

constexpr float kMinThreadHold = 0.1;

using ::mediapipe::NormalizedRect;
using ::mediapipe::api3::GenericGraph;
using ::mediapipe::api3::Packet;
using ::mediapipe::api3::Runner;
using ::mediapipe::api3::Stream;
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
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner,
      Runner::For([&](GenericGraph& graph,
                      Stream<std::vector<NormalizedRect>> base_rects,
                      Stream<std::vector<NormalizedRect>> rects0,
                      Stream<std::vector<NormalizedRect>> rects1)
                      -> Stream<std::vector<NormalizedRect>> {
        auto& node = graph.AddNode<tasks::HandAssociationNode>();
        {
          mediapipe::HandAssociationCalculatorOptions& options =
              *node.options.Mutable();
          options.set_min_similarity_threshold(kMinThreadHold);
        }
        node.base_rects.Add(base_rects);
        node.rects.Add(rects0);
        node.rects.Add(rects1);
        return node.output_rects.Get();
      }).Create());

  // Input Stream 0: nr_0, nr_1, nr_2.
  auto input_vec_0 = std::vector<NormalizedRect>();
  input_vec_0.push_back(nr_0_);
  input_vec_0.push_back(nr_1_);
  input_vec_0.push_back(nr_2_);

  // Input Stream 1: nr_3, nr_4.
  auto input_vec_1 = std::vector<NormalizedRect>();
  input_vec_1.push_back(nr_3_);
  input_vec_1.push_back(nr_4_);

  // Input Stream 2: nr_5.
  auto input_vec_2 = std::vector<NormalizedRect>();
  input_vec_2.push_back(nr_5_);

  MP_ASSERT_OK_AND_ASSIGN(
      Packet<std::vector<NormalizedRect>> output_packet,
      runner.Run(
          api3::MakePacket<std::vector<NormalizedRect>>(std::move(input_vec_0)),
          api3::MakePacket<std::vector<NormalizedRect>>(std::move(input_vec_1)),
          api3::MakePacket<std::vector<NormalizedRect>>(
              std::move(input_vec_2))));
  ASSERT_TRUE(output_packet);
  const std::vector<NormalizedRect>& assoc_rects = output_packet.GetOrDie();
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
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner,
      Runner::For([&](GenericGraph& graph,
                      Stream<std::vector<NormalizedRect>> base_rects,
                      Stream<std::vector<NormalizedRect>> rects0)
                      -> Stream<std::vector<NormalizedRect>> {
        auto& node = graph.AddNode<tasks::HandAssociationNode>();
        node.options.Mutable()->set_min_similarity_threshold(kMinThreadHold);
        node.base_rects.Add(base_rects);
        node.rects.Add(rects0);
        return node.output_rects.Get();
      }).Create());

  // Input Stream 0: nr_0, nr_1.  Tracked hands.
  auto input_vec_0 = std::vector<NormalizedRect>();
  nr_0_.set_rect_id(-2);
  input_vec_0.push_back(nr_0_);
  nr_1_.set_rect_id(-1);
  input_vec_0.push_back(nr_1_);

  // Input Stream 1: nr_2, nr_3. Newly detected palms.
  auto input_vec_1 = std::vector<NormalizedRect>();
  input_vec_1.push_back(nr_2_);
  input_vec_1.push_back(nr_3_);

  MP_ASSERT_OK_AND_ASSIGN(
      Packet<std::vector<NormalizedRect>> output_packet,
      runner.Run(
          api3::MakePacket<std::vector<NormalizedRect>>(std::move(input_vec_0)),
          api3::MakePacket<std::vector<NormalizedRect>>(
              std::move(input_vec_1))));
  ASSERT_TRUE(output_packet);
  const std::vector<NormalizedRect>& assoc_rects = output_packet.GetOrDie();

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
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner,
      Runner::For([&](GenericGraph& graph,
                      Stream<std::vector<NormalizedRect>> base_rects,
                      Stream<std::vector<NormalizedRect>> rects0,
                      Stream<std::vector<NormalizedRect>> rects1)
                      -> Stream<std::vector<NormalizedRect>> {
        auto& node = graph.AddNode<tasks::HandAssociationNode>();
        node.options.Mutable()->set_min_similarity_threshold(kMinThreadHold);
        node.base_rects.Add(base_rects);
        node.rects.Add(rects0);
        node.rects.Add(rects1);
        return node.output_rects.Get();
      }).Create());

  // Input Stream 0: nr_5
  auto input_vec_0 = std::vector<NormalizedRect>();
  input_vec_0.push_back(nr_5_);

  // Input Stream 1: nr_4, nr_3
  auto input_vec_1 = std::vector<NormalizedRect>();
  input_vec_1.push_back(nr_4_);
  input_vec_1.push_back(nr_3_);

  // Input Stream 2: nr_2, nr_1, nr_0
  auto input_vec_2 = std::vector<NormalizedRect>();
  input_vec_2.push_back(nr_2_);
  input_vec_2.push_back(nr_1_);
  input_vec_2.push_back(nr_0_);

  MP_ASSERT_OK_AND_ASSIGN(
      Packet<std::vector<NormalizedRect>> output_packet,
      runner.Run(
          api3::MakePacket<std::vector<NormalizedRect>>(std::move(input_vec_0)),
          api3::MakePacket<std::vector<NormalizedRect>>(std::move(input_vec_1)),
          api3::MakePacket<std::vector<NormalizedRect>>(
              std::move(input_vec_2))));
  ASSERT_TRUE(output_packet);
  const std::vector<NormalizedRect>& assoc_rects = output_packet.GetOrDie();

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
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner,
      Runner::For([&](GenericGraph& graph,
                      Stream<std::vector<NormalizedRect>> base_rects,
                      Stream<std::vector<NormalizedRect>> rects0,
                      Stream<std::vector<NormalizedRect>> rects1)
                      -> Stream<std::vector<NormalizedRect>> {
        auto& node = graph.AddNode<tasks::HandAssociationNode>();
        node.options.Mutable()->set_min_similarity_threshold(kMinThreadHold);
        node.base_rects.Add(base_rects);
        node.rects.Add(rects0);
        node.rects.Add(rects1);
        return node.output_rects.Get();
      }).Create());

  // Input Stream 0: nr_5, nr_3, nr_1.
  auto input_vec_0 = std::vector<NormalizedRect>();
  input_vec_0.push_back(nr_5_);
  input_vec_0.push_back(nr_3_);
  input_vec_0.push_back(nr_1_);

  // Input Stream 1: nr_4.
  auto input_vec_1 = std::vector<NormalizedRect>();
  input_vec_1.push_back(nr_4_);

  // Input Stream 2: nr_2, nr_0.
  auto input_vec_2 = std::vector<NormalizedRect>();
  input_vec_2.push_back(nr_2_);
  input_vec_2.push_back(nr_0_);

  MP_ASSERT_OK_AND_ASSIGN(
      Packet<std::vector<NormalizedRect>> output_packet,
      runner.Run(
          api3::MakePacket<std::vector<NormalizedRect>>(std::move(input_vec_0)),
          api3::MakePacket<std::vector<NormalizedRect>>(std::move(input_vec_1)),
          api3::MakePacket<std::vector<NormalizedRect>>(
              std::move(input_vec_2))));
  ASSERT_TRUE(output_packet);
  const std::vector<NormalizedRect>& assoc_rects = output_packet.GetOrDie();

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
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner,
      Runner::For([&](GenericGraph& graph,
                      Stream<std::vector<NormalizedRect>> base_rects)
                      -> Stream<std::vector<NormalizedRect>> {
        auto& node = graph.AddNode<tasks::HandAssociationNode>();
        node.options.Mutable()->set_min_similarity_threshold(kMinThreadHold);
        node.base_rects.Add(base_rects);
        return node.output_rects.Get();
      }).Create());

  // Just one input stream : nr_3, nr_5.
  auto input_vec = std::vector<NormalizedRect>();
  input_vec.push_back(nr_3_);
  input_vec.push_back(nr_5_);

  MP_ASSERT_OK_AND_ASSIGN(
      Packet<std::vector<NormalizedRect>> output_packet,
      runner.Run(
          api3::MakePacket<std::vector<NormalizedRect>>(std::move(input_vec))));
  ASSERT_TRUE(output_packet);
  const std::vector<NormalizedRect>& assoc_rects = output_packet.GetOrDie();

  // Rectangles are added in the following sequence:
  // nr_3 is added 1st.
  // nr_5 is added 2nd. The calculator assumes it does not overlap with nr_3.
  EXPECT_EQ(2, assoc_rects.size());

  nr_3_.set_rect_id(1);
  nr_5_.set_rect_id(2);
  EXPECT_THAT(assoc_rects, ElementsAre(EqualsProto(nr_3_), EqualsProto(nr_5_)));
}

TEST_F(HandAssociationCalculatorTest, HasCorrectRegistrationName) {
  EXPECT_EQ(tasks::HandAssociationNode::GetRegistrationName(),
            "HandAssociationCalculator");
}

}  // namespace
}  // namespace mediapipe
