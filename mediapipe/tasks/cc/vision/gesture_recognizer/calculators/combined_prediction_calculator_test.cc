/* Copyright 2025 The MediaPipe Authors.

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
#include "mediapipe/tasks/cc/vision/gesture_recognizer/calculators/combined_prediction_calculator.h"

#include <memory>
#include <string>
#include <utility>

#include "mediapipe/framework/api3/function_runner.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/packet.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

namespace {

using ::mediapipe::api3::GenericGraph;
using ::mediapipe::api3::Packet;
using ::mediapipe::api3::Runner;
using ::mediapipe::api3::Stream;

std::unique_ptr<ClassificationList> BuildCustomScoreInput(
    const float negative_score, const float drama_score,
    const float llama_score) {
  auto custom_scores = std::make_unique<ClassificationList>();
  auto custom_negative = custom_scores->add_classification();
  custom_negative->set_label("Negative");
  custom_negative->set_score(negative_score);
  auto drama = custom_scores->add_classification();
  drama->set_label("CustomDrama");
  drama->set_score(drama_score);
  auto llama = custom_scores->add_classification();
  llama->set_label("CustomLlama");
  llama->set_score(llama_score);
  return custom_scores;
}

std::unique_ptr<ClassificationList> BuildCannedScoreInput(
    const float negative_score, const float bazinga_score,
    const float joy_score, const float peace_score) {
  auto canned_scores = std::make_unique<ClassificationList>();
  auto canned_negative = canned_scores->add_classification();
  canned_negative->set_label("Negative");
  canned_negative->set_score(negative_score);
  auto bazinga = canned_scores->add_classification();
  bazinga->set_label("CannedBazinga");
  bazinga->set_score(bazinga_score);
  auto joy = canned_scores->add_classification();
  joy->set_label("CannedJoy");
  joy->set_score(joy_score);
  auto peace = canned_scores->add_classification();
  peace->set_label("CannedPeace");
  peace->set_score(peace_score);
  return canned_scores;
}

void AddClass(const std::string& label, float threshold,
              mediapipe::CombinedPredictionCalculatorOptions& opts) {
  auto& c = *opts.add_class_();
  c.set_label(label);
  c.set_score_threshold(threshold);
}

TEST(CombinedPredictionCalculatorPacketTest,
     CustomEmpty_CannedEmpty_ResultIsEmpty) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, Runner::For([](GenericGraph& graph,
                                  Stream<ClassificationList> custom_scores,
                                  Stream<ClassificationList> canned_scores)
                                   -> Stream<ClassificationList> {
                     auto& node =
                         graph.AddNode<tasks::CombinedPredictionNode>();
                     node.classification_list_in.Add(custom_scores);
                     node.classification_list_in.Add(canned_scores);
                     mediapipe::CombinedPredictionCalculatorOptions& opts =
                         *node.options.Mutable();
                     AddClass("CustomDrama", 0.0, opts);
                     AddClass("CustomLlama", 0.0, opts);
                     AddClass("CannedBazinga", 0.0, opts);
                     AddClass("CannedJoy", 0.0, opts);
                     AddClass("CannedPeace", 0.0, opts);
                     opts.set_background_label("Negative");
                     return node.prediction_out.Get();
                   }).Create());

  MP_ASSERT_OK_AND_ASSIGN(
      Packet<ClassificationList> output_packet,
      runner.Run(Packet<ClassificationList>(), Packet<ClassificationList>()));
  EXPECT_FALSE(output_packet);
}

TEST(CombinedPredictionCalculatorPacketTest,
     CustomEmpty_CannedNotEmpty_ResultIsCanned) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, Runner::For([](GenericGraph& graph,
                                  Stream<ClassificationList> custom_scores,
                                  Stream<ClassificationList> canned_scores)
                                   -> Stream<ClassificationList> {
                     auto& node =
                         graph.AddNode<tasks::CombinedPredictionNode>();
                     node.classification_list_in.Add(custom_scores);
                     node.classification_list_in.Add(canned_scores);
                     mediapipe::CombinedPredictionCalculatorOptions& opts =
                         *node.options.Mutable();
                     AddClass("CustomDrama", 0.0, opts);
                     AddClass("CustomLlama", 0.0, opts);
                     AddClass("CannedBazinga", 0.9, opts);
                     AddClass("CannedJoy", 0.5, opts);
                     AddClass("CannedPeace", 0.8, opts);
                     opts.set_background_label("Negative");
                     return node.prediction_out.Get();
                   }).Create());

  auto canned_scores = BuildCannedScoreInput(
      /*negative_score=*/0.1,
      /*bazinga_score=*/0.1, /*joy_score=*/0.6, /*peace_score=*/0.2);

  MP_ASSERT_OK_AND_ASSIGN(Packet<ClassificationList> output_packet,
                          runner.Run(Packet<ClassificationList>(),
                                     api3::MakePacket<ClassificationList>(
                                         std::move(canned_scores))));

  ASSERT_TRUE(output_packet);
  const ClassificationList& output_prediction_list = output_packet.GetOrDie();
  ASSERT_EQ(output_prediction_list.classification_size(), 1);
  Classification output_prediction = output_prediction_list.classification(0);

  EXPECT_EQ(output_prediction.label(), "CannedJoy");
  EXPECT_NEAR(output_prediction.score(), 0.6, 1e-4);
}

TEST(CombinedPredictionCalculatorPacketTest,
     CustomNotEmpty_CannedEmpty_ResultIsCustom) {
  MP_ASSERT_OK_AND_ASSIGN(
      auto runner, Runner::For([](GenericGraph& graph,
                                  Stream<ClassificationList> custom_scores,
                                  Stream<ClassificationList> canned_scores)
                                   -> Stream<ClassificationList> {
                     auto& node =
                         graph.AddNode<tasks::CombinedPredictionNode>();
                     node.classification_list_in.Add(custom_scores);
                     node.classification_list_in.Add(canned_scores);
                     mediapipe::CombinedPredictionCalculatorOptions& opts =
                         *node.options.Mutable();
                     AddClass("CustomDrama", 0.3, opts);
                     AddClass("CustomLlama", 0.5, opts);
                     AddClass("CannedBazinga", 0.0, opts);
                     AddClass("CannedJoy", 0.0, opts);
                     AddClass("CannedPeace", 0.0, opts);
                     opts.set_background_label("Negative");
                     return node.prediction_out.Get();
                   }).Create());

  auto custom_scores =
      BuildCustomScoreInput(/*negative_score=*/0.1,
                            /*drama_score=*/0.2, /*llama_score=*/0.7);
  MP_ASSERT_OK_AND_ASSIGN(
      Packet<ClassificationList> output_packet,
      runner.Run(api3::MakePacket<ClassificationList>(std::move(custom_scores)),
                 Packet<ClassificationList>()));

  ASSERT_TRUE(output_packet);
  const ClassificationList& output_prediction_list = output_packet.GetOrDie();
  ASSERT_EQ(output_prediction_list.classification_size(), 1);
  Classification output_prediction = output_prediction_list.classification(0);

  EXPECT_EQ(output_prediction.label(), "CustomLlama");
  EXPECT_NEAR(output_prediction.score(), 0.7, 1e-4);
}

struct CombinedPredictionCalculatorTestCase {
  std::string test_name;
  float custom_negative_score;
  float drama_score;
  float llama_score;
  float drama_thresh;
  float llama_thresh;
  float canned_negative_score;
  float bazinga_score;
  float joy_score;
  float peace_score;
  float bazinga_thresh;
  float joy_thresh;
  float peace_thresh;
  std::string max_scoring_label;
  float max_score;
};

using CombinedPredictionCalculatorTest =
    testing::TestWithParam<CombinedPredictionCalculatorTestCase>;

TEST_P(CombinedPredictionCalculatorTest, OutputsCorrectResult) {
  const CombinedPredictionCalculatorTestCase& test_params = GetParam();

  MP_ASSERT_OK_AND_ASSIGN(
      auto runner,
      Runner::For([&](GenericGraph& graph,
                      Stream<ClassificationList> custom_scores,
                      Stream<ClassificationList> canned_scores)
                      -> Stream<ClassificationList> {
        auto& node = graph.AddNode<tasks::CombinedPredictionNode>();
        node.classification_list_in.Add(custom_scores);
        node.classification_list_in.Add(canned_scores);
        mediapipe::CombinedPredictionCalculatorOptions& opts =
            *node.options.Mutable();
        AddClass("CustomDrama", test_params.drama_thresh, opts);
        AddClass("CustomLlama", test_params.llama_thresh, opts);
        AddClass("CannedBazinga", test_params.bazinga_thresh, opts);
        AddClass("CannedJoy", test_params.joy_thresh, opts);
        AddClass("CannedPeace", test_params.peace_thresh, opts);
        opts.set_background_label("Negative");
        return node.prediction_out.Get();
      }).Create());

  auto custom_scores =
      BuildCustomScoreInput(test_params.custom_negative_score,
                            test_params.drama_score, test_params.llama_score);

  auto canned_scores = BuildCannedScoreInput(
      test_params.canned_negative_score, test_params.bazinga_score,
      test_params.joy_score, test_params.peace_score);

  MP_ASSERT_OK_AND_ASSIGN(
      Packet<ClassificationList> output_packet,
      runner.Run(
          api3::MakePacket<ClassificationList>(std::move(custom_scores)),
          api3::MakePacket<ClassificationList>(std::move(canned_scores))));

  ASSERT_TRUE(output_packet);
  const ClassificationList& output_prediction_list = output_packet.GetOrDie();
  ASSERT_EQ(output_prediction_list.classification_size(), 1);
  Classification output_prediction = output_prediction_list.classification(0);

  EXPECT_EQ(output_prediction.label(), test_params.max_scoring_label);
  EXPECT_NEAR(output_prediction.score(), test_params.max_score, 1e-4);
}

INSTANTIATE_TEST_CASE_P(
    CombinedPredictionCalculatorTests, CombinedPredictionCalculatorTest,
    testing::ValuesIn<CombinedPredictionCalculatorTestCase>({
        {
            /* test_name= */ "TestCustomDramaWinnnerWith_HighCanned_Thresh",
            /* custom_negative_score= */ 0.1,
            /* drama_score= */ 0.5,
            /* llama_score= */ 0.3,
            /* drama_thresh= */ 0.25,
            /* llama_thresh= */ 0.7,
            /* canned_negative_score= */ 0.1,
            /* bazinga_score= */ 0.3,
            /* joy_score= */ 0.3,
            /* peace_score= */ 0.3,
            /* bazinga_thresh= */ 0.7,
            /* joy_thresh= */ 0.7,
            /* peace_thresh= */ 0.7,
            /* max_scoring_label= */ "CustomDrama",
            /* max_score= */ 0.5,
        },
        {
            /* test_name= */ "TestCannedWinnerWith_HighCustom_ZeroCanned_"
                             "Thresh",
            /* custom_negative_score= */ 0.1,
            /* drama_score= */ 0.3,
            /* llama_score= */ 0.6,
            /* drama_thresh= */ 0.4,
            /* llama_thresh= */ 0.8,
            /* canned_negative_score= */ 0.1,
            /* bazinga_score= */ 0.4,
            /* joy_score= */ 0.3,
            /* peace_score= */ 0.2,
            /* bazinga_thresh= */ 0.0,
            /* joy_thresh= */ 0.0,
            /* peace_thresh= */ 0.0,
            /* max_scoring_label= */ "CannedBazinga",
            /* max_score= */ 0.4,
        },
        {
            /* test_name= */ "TestNegativeWinnerWith_LowCustom_HighCanned_"
                             "Thresh",
            /* custom_negative_score= */ 0.5,
            /* drama_score= */ 0.1,
            /* llama_score= */ 0.4,
            /* drama_thresh= */ 0.1,
            /* llama_thresh= */ 0.05,
            /* canned_negative_score= */ 0.1,
            /* bazinga_score= */ 0.3,
            /* joy_score= */ 0.3,
            /* peace_score= */ 0.3,
            /* bazinga_thresh= */ 0.7,
            /* joy_thresh= */ 0.7,
            /* peace_thresh= */ 0.7,
            /* max_scoring_label= */ "Negative",
            /* max_score= */ 0.5,
        },
        {
            /* test_name= */ "TestNegativeWinnerWith_HighCustom_HighCanned_"
                             "Thresh",
            /* custom_negative_score= */ 0.8,
            /* drama_score= */ 0.1,
            /* llama_score= */ 0.1,
            /* drama_thresh= */ 0.25,
            /* llama_thresh= */ 0.7,
            /* canned_negative_score= */ 0.1,
            /* bazinga_score= */ 0.3,
            /* joy_score= */ 0.3,
            /* peace_score= */ 0.3,
            /* bazinga_thresh= */ 0.7,
            /* joy_thresh= */ 0.7,
            /* peace_thresh= */ 0.7,
            /* max_scoring_label= */ "Negative",
            /* max_score= */ 0.8,
        },
        {
            /* test_name= */ "TestNegativeWinnerWith_HighCustom_"
                             "HighCannedThresh2",
            /* custom_negative_score= */ 0.1,
            /* drama_score= */ 0.2,
            /* llama_score= */ 0.7,
            /* drama_thresh= */ 1.1,
            /* llama_thresh= */ 1.1,
            /* canned_negative_score= */ 0.1,
            /* bazinga_score= */ 0.3,
            /* joy_score= */ 0.3,
            /* peace_score= */ 0.3,
            /* bazinga_thresh= */ 0.7,
            /* joy_thresh= */ 0.7,
            /* peace_thresh= */ 0.7,
            /* max_scoring_label= */ "Negative",
            /* max_score= */ 0.1,
        },
        {
            /* test_name= */ "TestNegativeWinnerWith_HighCustom_HighCanned_"
                             "Thresh3",
            /* custom_negative_score= */ 0.1,
            /* drama_score= */ 0.3,
            /* llama_score= */ 0.6,
            /* drama_thresh= */ 0.4,
            /* llama_thresh= */ 0.8,
            /* canned_negative_score= */ 0.3,
            /* bazinga_score= */ 0.2,
            /* joy_score= */ 0.3,
            /* peace_score= */ 0.2,
            /* bazinga_thresh= */ 0.5,
            /* joy_thresh= */ 0.5,
            /* peace_thresh= */ 0.5,
            /* max_scoring_label= */ "Negative",
            /* max_score= */ 0.1,
        },
    }),
    [](const testing::TestParamInfo<
        CombinedPredictionCalculatorTest::ParamType>& info) {
      return info.param.test_name;
    });

TEST(CombinedPredictionCalculatorTest, HasCorrectRegistrationName) {
  EXPECT_EQ(tasks::CombinedPredictionNode::GetRegistrationName(),
            "CombinedPredictionCalculator");
}

}  // namespace

}  // namespace mediapipe
